#%%
from collections import defaultdict
import os.path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle

import logging
from model import ResNet_cifar, Resnet_LT
from Trainer import Trainer
from imbalance_data import cifar100Imbanlance, dataset_lt_data, cifar10Imbanlance
from torchvision import transforms
from utils import util
import os
import numpy as np
import torch
from torch.backends import cudnn
import random
import argparse
import time

dataset = 'iNaturelist2018'
#dataset = 'imagenet'
dataset = 'cifar100'

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


args.root = 'data/'
args.print_freq = 100
args.store_name = ''

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


if dataset == 'cifar10':
    #  --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1
    args.dataset = 'cifar10'
    args.arch='resnet32'
    args.num_classes = 10
    args.imbanlance_rate = 0.01
    args.batch_size = 64
    args.resample_weighting = 0.0
    args.beta = 0.5
    args.lr = 0.01
    args.epochs = 200
    args.momentum = 0.9
    args.weight_decay = 5e-3
    args.label_weighting = 1.2
    args.contrast_weight = 1
    
    coarse_labels = np.array([0]*10)
    coarse_labels = np.array([0, 1, 0, 2, 2, 2, 2, 2, 0, 1])

                             
    model = ResNet_cifar.resnet32(num_class=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    
    args.resume = os.path.join('GLMC-CVPR2023', 'output', 'cifar10_ckpt.best.pth.tar')
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        
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
    
    train_dataset = cifar10Imbanlance.Cifar10Imbanlance(transform=util.TwoCropTransform(transform_train),imbanlance_rate=args.imbanlance_rate, train=True,file_path=args.root)
    val_dataset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=args.root)

if dataset == 'cifar100':
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
        
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    
    model = ResNet_cifar.resnet32(num_class=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    
    args.resume = os.path.join('GLMC-CVPR2023', 'output', 'cifar100_ckpt.best.pth.tar')
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    
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

if dataset == 'iNaturelist2018':
    #--dataset iNaturelist2018 -a resnext50_32x4d --num_classes 8142 --beta 0.5 --lr 0.1 --epochs 120 -b 128 --momentum 0.9 --weight_decay 1e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
    args.dataset = 'iNaturelist2018'
    args.arch    = 'resnext50_32x4d'
    args.num_classes = 8142
    args.beta = 0.5
    args.lr = 0.1
    args.epochs = 120
    args.batch_size = 128
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.label_weighting = 1
    args.contrast_weight = 10
    args.resample_weighting = 0.2
    
    model =  Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    
    transform_train,transform_val = util.get_transform(args.dataset)
    train_dataset = dataset_lt_data.LT_Dataset(args.root, 'data/data_txt/iNaturalist18_train.txt',util.TwoCropTransform(transform_train))
    val_dataset = dataset_lt_data.LT_Dataset(args.root, 'data/data_txt/iNaturalist18_val.txt',transform_val)

if dataset == 'ImageNet-LT':
    #--dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --beta 0.5 --lr 0.1 --epochs 135 -b 120 --momentum 0.9 --weight_decay 2e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
    args.dataset = 'ImageNet-LT'
    args.arch    = 'resnext50_32x4d'
    args.num_classes = 1000
    args.beta = 0.5
    args.lr = 0.1
    args.epochs = 135
    args.batch_size = 120
    args.momentum = 0.9
    args.weight_decay = 2e-4
    args.label_weighting = 1
    args.contrast_weight = 10
    args.resample_weighting = 0.2
    
    args.root = 'data/imagenet-lt/'
    
    model =  Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    
    transform_train,transform_val = util.get_transform(args.dataset)
    train_dataset = dataset_lt_data.LT_Dataset(args.root, 'data/data_txt/ImageNet_LT_train.txt',util.TwoCropTransform(transform_train))
    val_dataset = dataset_lt_data.LT_Dataset(args.root, 'data/data_txt/ImageNet_LT_test.txt',transform_val)


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
trainer = Trainer(args, model=model,train_loader=train_loader, val_loader=val_loader,weighted_train_loader=weighted_train_loader, per_class_num=train_cls_num_list, log=logging)


import torch.nn.functional as F


def model_get_features(model, x):
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.maxpool(out) # comment for cifar100
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.layer4(out)
    out = model.avgpool(out)
    #out = F.avg_pool2d(out, out.size()[3])
    feature = out.view(out.size(0), -1)
    return feature


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_coarse_labels = max(coarse_labels)+1
fine_labels = [list(np.where(coarse_labels == i)[0])
               for i in range(num_coarse_labels)]


#%% Load checkpoint
args.resume = os.path.join('GLMC-CVPR2023', 'output', 'cifar10_ckpt.best.pth.tar')
if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume, map_location='cuda:0')
    args.start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    if args.gpu is not None:
        # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(args.gpu)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume))
trainer.model = model
#%% Train a new model

start_time = time.time()
print("Training started!")
trainer.train()
end_time = time.time()
print("It took {} to execute the program".format(hms_string(end_time - start_time)))

#%% Create KZ Dataset
if dataset == 'cifar10':
    train_dataset = cifar10Imbanlance.Cifar10Imbanlance(transform=transform_val,imbanlance_rate=args.imbanlance_rate, train=True,file_path=args.root)
    
if dataset == 'cifar100':
    train_dataset = cifar100Imbanlance.Cifar100Imbanlance(transform=transform_val,
                                                          imbanlance_rate=args.imbanlance_rate,
                                                          train=True,
                                                          file_path=os.path.join('data/','cifar-100-python/')
                                                          )

if dataset == 'ImageNet-LT':
    train_dataset = dataset_lt_data.LT_Dataset(args.root, 'data/data_txt/ImageNet_LT_train.txt',transform_val)
    
    
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=(train_sampler is None),
                                           num_workers=4,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           sampler=train_sampler)

feature_dim = model.module.fc_cb.in_features

fc = model.module.fc_cb
#fc = pretrained_model.fc[1]

model.eval()
# Get KZ
with torch.no_grad():
    #model = model.cpu()
    kz_tensor_for_coarse = defaultdict(lambda: torch.zeros(0, feature_dim))
    pi_tensor_for_coarse = defaultdict(lambda: torch.zeros(0, feature_dim))
    kz_tensor_for_class = defaultdict(lambda: torch.zeros(0, feature_dim))
    pi_tensor_for_class = defaultdict(lambda: torch.zeros(0, feature_dim))
    targets_for_coarse = defaultdict(list)

    for i in range(len(train_dataset)):
        data, target = train_dataset[i]
        data = data.to(device).unsqueeze(0)
        coarse_label = coarse_labels[target]    
        
        _, _, _, _, features = model.module(data, train=True, extract_features=True)
        #features = pretrained_get_features(pretrained_model, data)
        outputs = fc(features)
        
        pi = (torch.linalg.pinv(fc.weight.data) @ (outputs - fc.bias.data).T).T
        kz = features - pi
        
        kz_tensor_for_coarse[coarse_label] = np.concatenate((kz_tensor_for_coarse[coarse_label], kz.cpu().numpy()), axis=0)
        pi_tensor_for_coarse[coarse_label] = np.concatenate((pi_tensor_for_coarse[coarse_label], pi.cpu().numpy()), axis=0)
        kz_tensor_for_class[target] = np.concatenate((kz_tensor_for_class[target], kz.cpu().numpy()), axis=0)
        pi_tensor_for_class[target] = np.concatenate((pi_tensor_for_class[target], pi.cpu().numpy()), axis=0)
        
        targets_for_coarse[coarse_label] += [target]
    model.to(device)
#%% Find top directions in KZ
directions_num = 5
def get_pca_directions(X, k):
    # X: n x d data matrix
    X_centered = X - X.mean(axis=0)  # Center the data
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    top_directions = eigenvectors[:, idx[:k]]  # Each column is a principal direction
    return top_directions.T, eigenvalues[idx[:k]]
top_directions_for_coarse = {}
for coarse_label in range(num_coarse_labels):
    top_directions, top_ev = get_pca_directions(kz_tensor_for_coarse[coarse_label], directions_num)
    top_directions_for_coarse[coarse_label] = top_directions
    
top_directions_all, top_ev_all = get_pca_directions(np.concatenate(list(kz_tensor_for_coarse.values()), axis=0), directions_num)

#%% Train classier using augmented data


class ImbanlanceAugmented(Dataset):
    def __init__(self, pi, kz, targets, fine_to_coarse, top_dir=None):
        self.kz_dataset = kz
        self.pi_dataset = pi
        self.targets = targets
        self.fine_to_coarse = fine_to_coarse
        num_coarse_labels = max(fine_to_coarse) + 1
        self.coarse_to_fine = [list(np.where(fine_to_coarse==i)[0]) for i in range(num_coarse_labels)]
        self.feat_dim = pi[0][0].shape[0]
        #self.orig_per_class_num = [self.targets[coarse].count(fine) for fine, coarse in enumerate(self.fine_to_coarse)]
        self.orig_per_class_num = [self.pi_dataset[c].shape[0] for c in self.pi_dataset.keys()]
        self.top_dir = top_dir
        
        self.kz_info = {}
        for c, X in self.kz_dataset.items():
            mean = X.mean(dim=0)
            X_centered = X - mean
            cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            first_pc = eigenvectors[:, -1]  # shape: (d,)
            first_eigenvalue = eigenvalues[-1]
            
            self.kz_info[c] = {'mean': mean,
                               'first_pc': first_pc,
                               'std': first_eigenvalue.sqrt(),
                               }
            
    
    def get_num_classes(self):
        return len(self.fine_to_coarse)
    
    def get_num_per_class(self):
        return max(self.orig_per_class_num)
    
    def __len__(self):
        return self.get_num_per_class() * self.get_num_classes()
    
    def __getitem__(self, item):
        y = item // self.get_num_per_class()
        idx_in_class = item % self.get_num_per_class()
        if idx_in_class < self.orig_per_class_num[y]:
            x = self.kz_dataset[y][idx_in_class, :] + self.pi_dataset[y][idx_in_class, :]
        else:
            # get random sample from said class
            pi_idx_in_class = idx_in_class % self.orig_per_class_num[y]  
            pi = self.pi_dataset[y][pi_idx_in_class, :]
            
            coarse_label = self.fine_to_coarse[y]
             
            if self.top_dir is None:  
                kz_class_options = self.coarse_to_fine[coarse_label]
                kz_class_numbers = [self.kz_dataset[c].shape[0] for c in kz_class_options]
                kz_class_probs = np.array(kz_class_numbers) / sum(kz_class_numbers)
                kz_class = np.random.choice(kz_class_options, p=kz_class_probs)
                kz_idx_in_class = np.random.randint(self.kz_dataset[kz_class].shape[0])
                kz = self.kz_dataset[kz_class][kz_idx_in_class, :]
                
                # project kz into y's pc
                kz = (kz - self.kz_info[kz_class]['mean'] +  self.kz_info[y]['mean']) * (self.kz_info[y]['std'] / self.kz_info[kz_class]['std'])

            else:
                kz = np.random.rand(self.top_dir[coarse_label].shape[0]) @ self.top_dir[coarse_label]
            #kz = self.kz_dataset[y][pi_idx_in_class, :] # Turns this into simple over-sampling dataset
            
            x = kz + pi
        return x.squeeze().detach(), y
    

class OversampledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
        self.class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.class_to_indices[label].append(idx)
    
        self.max_class_size = max(len(indices) for indices in self.class_to_indices.values())
        self.total_len = self.max_class_size * len(self.class_to_indices.keys())
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        class_id = item // self.max_class_size
        position = item % self.max_class_size

        class_indices = self.class_to_indices[class_id]
        true_idx = class_indices[position % len(class_indices)]

        return self.dataset[true_idx]
        
augmented_dataset = ImbanlanceAugmented(pi={k:torch.tensor(v) for k,v in pi_tensor_for_class.items()},
                                        kz={k:torch.tensor(v) for k,v in kz_tensor_for_class.items()},
                                        targets=targets_for_coarse,
                                        #fine_to_coarse=coarse_labels,
                                        fine_to_coarse=clusters,
                                        #top_dir=top_directions_for_coarse
                                        )

augmented_loader = torch.utils.data.DataLoader(augmented_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           sampler=train_sampler)

oversampled_dataset = OversampledDataset(train_dataset)
oversampled_loader = torch.utils.data.DataLoader(oversampled_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           sampler=train_sampler)

num_classes = model.module.fc_cb.out_features
classifier = torch.nn.Linear(feature_dim, num_classes).to(device)
"""
classifier = torch.nn.Sequential(
                torch.nn.Linear(256, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 100)
            ).to(device)
        """

classifier.load_state_dict(model.module.fc_cb.state_dict())
with torch.no_grad():
    classifier.weight += 0.01 * torch.randn_like(classifier.weight)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), momentum=0.9, lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

best_acc = 0
model.eval()
for epoch in range(30):
    for batch_idx, (data, target) in enumerate(augmented_loader):
        classifier.train()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #_,_,_,_,data = model(data, train=True, extract_features=True)
        outputs = classifier(data)
        loss = criterion(outputs, target.to(torch.long))
        loss.backward()
        optimizer.step()
    
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx}/{len(augmented_loader)} "
                  f"({100. * batch_idx / len(augmented_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    
    scheduler.step()
    
    
    model.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.squeeze(0), target.squeeze(0)
            _, _, _, _, features = model(data, train=True, extract_features=True)
            output = classifier(features)
            loss = criterion(output, target.to(torch.long))
            test_loss += loss.item()
            test_total += 1
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
            
    test_loss = test_loss / test_total
    test_accuracy = correct / total
    print(f"epoch {epoch} | Test accuracy: {test_accuracy} | Test loss: {test_loss:.4f}")
    
    best_acc = max(best_acc, test_accuracy)

print(best_acc)

#%%
count = {i: 0 for i in range(100)}
for i in range(len(augmented_dataset)):
    data, target = augmented_dataset[i]
    data = data.unsqueeze(0).to(device)
    idx_in_class = i % augmented_dataset.get_num_per_class()
    if i < augmented_dataset.orig_per_class_num[target]:
        continue
    idx_of_pi = idx_in_class % augmented_dataset.orig_per_class_num[target]
    pi_data = augmented_dataset[i - (idx_in_class - idx_of_pi)][0].unsqueeze(0).to(device)
    current_correct = classifier(data).argmax() == target
    orig_correct = classifier(pi_data).argmax() == target
    if orig_correct != current_correct:
        count[target] += 1
    

#%% Test per class
criterion = torch.nn.CrossEntropyLoss()

model.eval()
classifier.eval()
test_loss = 0
correct = 0
coarse_correct = 0
total = 0
test_total = 0
correct_per_class = {d:0 for d in range(args.num_classes)}
total_per_class = {d:0 for d in range(args.num_classes)}
acc_per_class = {d:0 for d in range(args.num_classes)}
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        data, target = data.squeeze(0), target.squeeze(0)
        _, _, _, _, features = model(data, train=True, extract_features=True)
        output = model.module.fc_cb(features)
        output = classifier(features)
        loss = criterion(output, target.to(torch.long))
        test_loss += loss.item()
        test_total += 1
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        coarse_correct += torch.tensor(coarse_labels[pred.cpu().numpy()]).eq(torch.tensor(coarse_labels[target.cpu().numpy()]).view_as(pred)).sum().item()
        total += len(target)
        
        for d in range(args.num_classes):
            correct_per_class[d] += pred.eq(target.view_as(pred)).flatten()[target==d].sum().item()
            total_per_class[d] += (target==d).sum().item()
for d in range(args.num_classes):
    acc_per_class[d] = correct_per_class[d]/total_per_class[d]
acc_per_coarse = {}
for c in range(len(fine_labels)):
    acc_per_coarse[c] = [acc_per_class[f] for f in fine_labels[c]]
test_loss = test_loss / test_total
test_accuracy = correct / total
print()
print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")
print()
for d in range(0):
    print(f"Accuracy for class {d}: {correct_per_class[d]/total_per_class[d]}")
for c in range(len(fine_labels)):
    print(f"Accuracy for coarse {c}: {acc_per_coarse[c]}")
print()
print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")

#%% Fine tune for normality of Kz


def normality_criterion(x):
    """
    x: Tensor of shape (batch_size, features)
    Returns a loss scalar penalizing deviation from normal distribution.
    """
    # Flatten if needed
    x = x.view(x.size(0), -1)

    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6  # avoid div by zero
    x_norm = (x - mean) / std

    # Skewness = E[(x - mu)^3] / sigma^3
    skewness = ((x_norm ** 3).mean(dim=1)) ** 2  # square to make it positive

    # Kurtosis = E[(x - mu)^4] / sigma^4
    kurtosis = ((x_norm ** 4).mean(dim=1))
    excess_kurtosis = (kurtosis - 3) ** 2  # square deviation from normal kurtosis

    loss = (skewness + excess_kurtosis).mean()
    return loss

from utils.util import AverageMeter
from utils import util

args.lr = 1e-4
#args.batch_size = 60
trainer = Trainer(args, model=model,train_loader=train_loader, val_loader=val_loader,weighted_train_loader=weighted_train_loader, per_class_num=train_cls_num_list, log=logging)
best_dict = trainer.model.state_dict()

finetune_epochs = 15
best_acc1 = 0
for epoch in range(finetune_epochs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()
    end = time.time()
    weighted_train_loader = iter(trainer.weighted_train_loader)

    for i, (inputs, targets) in enumerate(trainer.train_loader):
        trainer.optimizer.zero_grad()
        #print('hi')
        input_org_1 = inputs[0]
        input_org_2 = inputs[1]
        target_org = targets
        #print('hi')
        try:
            input_invs, target_invs = next(weighted_train_loader)
        except:
            weighted_train_loader = iter(trainer.weighted_train_loader)
            input_invs, target_invs = next(weighted_train_loader)

        input_invs_1 = input_invs[0][:input_org_1.size()[0]]
        input_invs_2 = input_invs[1][:input_org_2.size()[0]]

        one_hot_org = torch.zeros(target_org.size(0), trainer.num_classes).scatter_(1, target_org.view(-1, 1), 1)
        one_hot_org_w = trainer.per_cls_weights.cpu() * one_hot_org
        one_hot_invs = torch.zeros(target_invs.size(0), trainer.num_classes).scatter_(1, target_invs.view(-1, 1), 1)
        one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
        one_hot_invs_w = trainer.per_cls_weights.cpu() * one_hot_invs

        input_org_1 = input_org_1.cuda()
        input_org_2 = input_org_2.cuda()
        input_invs_1 = input_invs_1.cuda()
        input_invs_2 = input_invs_2.cuda()

        one_hot_org = one_hot_org.cuda()
        one_hot_org_w = one_hot_org_w.cuda()
        one_hot_invs = one_hot_invs.cuda()
        one_hot_invs_w = one_hot_invs_w.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # Data augmentation
        lam = np.random.beta(trainer.beta, trainer.beta)

        mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(org1=input_org_1, org2=input_org_2,
                                                                                invs1=input_invs_1,
                                                                                invs2=input_invs_2,
                                                                                label_org=one_hot_org,
                                                                                label_invs=one_hot_invs,
                                                                                label_org_w=one_hot_org_w,
                                                                                label_invs_w=one_hot_invs_w)

        
        _, output_cb_1, z1, p1, features_1 = model(mix_x, train=True, extract_features=True)
        _, output_cb_2, z2, p2, features_2 = model(cut_x, train=True, extract_features=True)
        contrastive_loss = trainer.SimSiamLoss(p1, z2) + trainer.SimSiamLoss(p2, z1)
        del z1, p1, z2, p2
        torch.cuda.empty_cache()
        
        loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
        loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

        rebalance_loss = loss_mix_w + loss_cut_w
        
        pi_1 = (torch.linalg.pinv(model.module.fc_cb.weight.data) @ (output_cb_1 - model.module.fc_cb.bias.data).T).T
        kz_1 = features_1 - pi_1
        pi_2 = (torch.linalg.pinv(model.module.fc_cb.weight.data) @ (output_cb_2 - model.module.fc_cb.bias.data).T).T
        kz_2 = features_2 - pi_2
        normality_loss = normality_criterion(kz_1) + normality_criterion(kz_2)
                
        loss = rebalance_loss + trainer.contrast_weight * contrastive_loss + 0.3*normality_loss
        losses.update(loss.item(), inputs[0].size(0))

        # compute gradient and do SGD step
        loss.backward()
        trainer.optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % trainer.print_freq == 0:
            output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch + 1, finetune_epochs, i, len(trainer.train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))  # TODO
            print(output)
            # evaluate on validation set
    acc1 = trainer.validate(epoch=epoch)
    trainer.train_scheduler.step()
    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1,  best_acc1)
    output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
    print(output_best)
    if is_best:
        best_dict = trainer.model.state_dict()
    """
    save_checkpoint(trainer.args, {
        'epoch': epoch + 1,
        'state_dict': trainer.model.state_dict(),
        'best_acc1':  best_acc1,
    }, is_best, epoch + 1)
    """
model.load_state_dict(best_dict)

#%% FIne tune for center loss

class WeightedCenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim, class_weights=None, device='cpu', alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.alpha = alpha  # learning rate for center update

        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim).to(device), requires_grad=False)

        if class_weights is not None:
            self.class_weights = class_weights.to(device)
        else:
            self.class_weights = torch.ones(num_classes, device=device)

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]  # [B, feat_dim]
        loss = (features - centers_batch).pow(2).sum(dim=1)  # [B]

        weights = self.class_weights[labels]  # [B]
        weighted_loss = loss * weights

        # Manual center update (no gradient)
        with torch.no_grad():
            for i in range(batch_size):
                label = labels[i]
                diff = self.centers[label] - features[i]
                self.centers[label] -= self.alpha * diff

        return weighted_loss.mean()

if dataset == 'cifar10':
    train_dataset = cifar10Imbanlance.Cifar10Imbanlance(transform=transform_val,imbanlance_rate=args.imbanlance_rate, train=True,file_path=args.root)
    num_classes=10
    feat_dim=256
    
if dataset == 'cifar100':
    train_dataset = cifar100Imbanlance.Cifar100Imbanlance(transform=transform_val,
                                                          imbanlance_rate=args.imbanlance_rate,
                                                          train=True,
                                                          file_path=os.path.join('data/','cifar-100-python/')
                                                          )
    num_classes=100
    feat_dim=256

if dataset == 'ImageNet-LT':
    train_dataset = dataset_lt_data.LT_Dataset(args.root, 'data/data_txt/ImageNet_LT_train.txt',transform_val)
    
    
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=(train_sampler is None),
                                           num_workers=4,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           sampler=train_sampler)

class_weights = 1.0 / trainer.cls_num_list
class_weights = class_weights / np.sum(class_weights) * len(class_weights)
class_weights = torch.tensor(class_weights).to(device).to(torch.float)
center_loss_criterion = WeightedCenterLoss(num_classes, feat_dim, class_weights=class_weights, device=device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
lambda_center = 0.05

best_acc = 0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
num_epochs=15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        _, _, _, _, features = model.module(data, train=True, extract_features=True)
        outputs = model.module.fc_cb(features)
        pi = (torch.linalg.pinv(model.module.fc_cb.weight.data) @ (outputs - model.module.fc_cb.bias.data).T).T
        

        # Compute losses
        ce_loss = criterion(outputs, labels)
        center_loss = center_loss_criterion(pi, labels)

        # Combine
        total_loss = ce_loss + lambda_center * center_loss

        # Backward + optimize
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")
    
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.squeeze(0), target.squeeze(0)
            _, output, _, _, features = model(data, train=True, extract_features=True)
            loss = criterion(output, target.to(torch.long))
            test_loss += loss.item()
            test_total += 1
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
            
    test_loss = test_loss / test_total
    test_accuracy = correct / total
    print(f"epoch {epoch+1} | Test accuracy: {test_accuracy} | Test loss: {test_loss:.4f}")
    
    best_acc = max(best_acc, test_accuracy)