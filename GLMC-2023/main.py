
from model import ResNet_cifar
from Trainer import Trainer
from imbalance_data import cifar100Imbanlance
from torchvision import transforms
from utils import util
import os
import numpy as np
import torch
from torch.backends import cudnn
import random
import argparse
import time

parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
parser.add_argument('--root', type=str, default='/data/', help="dataset setting")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',choices=('resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'))
parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--beta', type=float, default=0.5, help="augment mixture")
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate',dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight_decay', default=5e-3, type=float, metavar='W',help='weight decay (default: 5e-3、2e-4、1e-4)', dest='weight_decay')
parser.add_argument('--resample_weighting', default=0.2, type=float,help='weighted for sampling probability (q(1,k))')
parser.add_argument('--label_weighting', default=1.0, type=float, help='weighted for Loss')
parser.add_argument('--contrast_weight', default=10,type=int,help='Mixture Consistency  Weights')
# etc.
parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
parser.add_argument('-p', '--print_freq', default=1000, type=int, metavar='N',help='print frequency (default: 100)')
parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--root_log', type=str, default='GLMC-CVPR2023/output/')
parser.add_argument('--root_model', type=str, default='GLMC-CVPR2023/output/')
parser.add_argument('--store_name', type=str, default='GLMC-CVPR2023/output/')
args = parser.parse_args()


# --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2  --contrast_weight 4
args.dataset = 'cifar100'
args.arch='resnet32'
args.num_classes = 100
args.imbanlance_rate = 0.01
args.batch_size = 64
args.resample_weighting = 0.0
args.beta = 0.5
args.lr = 0.01
args.epochs = 200
args.momentum = 0.9
args.weight_decay = 5e-3
args.label_weighting = 1.2
args.contrast_weight = 4
args.root = 'data/'
args.print_freq = 1


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

model = ResNet_cifar.resnet32(num_class=args.num_classes)
model = torch.nn.DataParallel(model).cuda()

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = cifar100Imbanlance.Cifar100Imbanlance(transform=util.TwoCropTransform(transform_train),
                                                      imbanlance_rate=args.imbanlance_rate,
                                                      train=True,
                                                      file_path=os.path.join('data/','cifar-100-python/')
                                                      )
val_dataset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate,
                                                    train=False,
                                                    transform=transform_val,
                                                    file_path=os.path.join('data/','cifar-100-python/')
                                                    )
assert args.num_classes == len(np.unique(train_dataset.targets))

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = True

cls_num_list = train_dataset.get_per_class_num()
train_sampler = None
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=(train_sampler is None),
                                           num_workers=4,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         persistent_workers=True,
                                         pin_memory=True)

cls_num_list = [0] * args.num_classes
for label in train_dataset.targets:
    cls_num_list[label] += 1
train_cls_num_list = np.array(cls_num_list)
train_sampler = None
weighted_train_loader = None


#weighted_loader
cls_weight = 1.0 / (np.array(cls_num_list) ** args.resample_weighting)
cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
weighted_train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    num_workers=4,
                                                    persistent_workers=True,
                                                    pin_memory=True,
                                                    sampler=weighted_sampler)

cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
start_time = time.time()
print("Training started!")
trainer = Trainer(args, model=model,train_loader=train_loader, val_loader=val_loader,weighted_train_loader=weighted_train_loader, per_class_num=train_cls_num_list)
trainer.train()
end_time = time.time()
print("It took {} to execute the program".format(hms_string(end_time - start_time)))
