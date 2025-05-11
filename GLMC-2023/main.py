
import logging
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
args.print_freq = 100
args.store_name = ''

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
trainer = Trainer(args, model=model,train_loader=train_loader, val_loader=val_loader,weighted_train_loader=weighted_train_loader, per_class_num=train_cls_num_list, log=logging)
#%% Load checkpoint
args.resume = os.path.join('GLMC-CVPR2023', 'output', 'ckpt.best.pth.tar')
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

#%% Train a new model

start_time = time.time()
print("Training started!")
trainer.train()
end_time = time.time()
print("It took {} to execute the program".format(hms_string(end_time - start_time)))

#%% Train classier using augmented data

import torch.nn.functional as F

def model_get_features(model, x):
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = F.avg_pool2d(out, out.size()[3])
    feature = out.view(out.size(0), -1)
    return feature


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#prev_classifier = model.module.fc_cb

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
train_dataset = cifar100Imbanlance.Cifar100Imbanlance(transform=transform_train,
                                                      imbanlance_rate=args.imbanlance_rate,
                                                      train=True,
                                                      file_path=os.path.join('data/','cifar-100-python/')
                                                      )

train_sampler = None
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=(train_sampler is None),
                                           num_workers=4,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           sampler=train_sampler)
# Get KZ
with torch.no_grad():
    classifier_to_use = model.module.fc_cb
    #classifier_to_use = classifier
    #model = model.cpu()
    kz_tensor = np.zeros((0,256))
    kz_targets_tensor = np.empty(0)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        features = model_get_features(model.module, data)
        outputs = classifier_to_use(features)
        pi = (torch.linalg.pinv(classifier_to_use.weight.data) @ (outputs - classifier_to_use.bias.data).T).T
        kz = features - pi
        kz_tensor = np.concatenate((kz_tensor, kz.cpu().numpy()), axis=0)
        kz_targets_tensor = np.concatenate((kz_targets_tensor, target.numpy()), axis=0)
        print(f'{batch_idx}/170')
    model.to(device)

# Find top directions in KZ
directions_num = 10
def get_pca_directions(X, k):
    # X: n x d data matrix
    X_centered = X - X.mean(axis=0)  # Center the data
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    top_directions = eigenvectors[:, idx[:k]]  # Each column is a principal direction
    return top_directions.T
top_directions = get_pca_directions(kz_tensor, directions_num)

#%%

import os.path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
from PIL import Image

from torchvision import transforms


class ImbanlanceAugmented(Dataset):
    def __init__(self, imbalanced_dataset, directions):
        self.imbalanced_dataset = imbalanced_dataset
        self.directions = directions
        ndir = self.directions.shape[0]
        self.direction_indicator = [(i,) for i in range(ndir)] + random.sample([(i, j) for i in range(ndir) for j in range(ndir) if i != j], k=ndir*(ndir-1))
    
    def get_num_classes(self):
        return len(self.imbalanced_dataset.per_class_num)
    
    def get_num_per_class(self):
        return self.imbalanced_dataset.per_class_num[0]
    
    def __len__(self):
        return self.get_num_per_class() * self.get_num_classes()
    
    def __getitem__(self, item):
        class_ = item // self.get_num_per_class()
        idx_in_class = item % self.get_num_per_class()
        if idx_in_class < self.imbalanced_dataset.per_class_num[class_]:
            x, y = self.imbalanced_dataset[sum(self.imbalanced_dataset.per_class_num[:class_]) + idx_in_class]
            aug = torch.zeros(self.directions.shape[1])
        else:
            # get random sample from said class
            sample_index = torch.randint(low=0, high=self.imbalanced_dataset.per_class_num[class_], size=(1,))
            x, y = self.imbalanced_dataset[sum(self.imbalanced_dataset.per_class_num[:class_]) + sample_index]
            # get augmentation
            
            direction_indicator_index = (idx_in_class - self.imbalanced_dataset.per_class_num[class_]) % len(self.direction_indicator)
            direction_index = self.direction_indicator[direction_indicator_index]
            if len(direction_index) == 1:
                weights = 1
            else:
                weights = torch.randn((len(direction_index),1))
                weights /= weights.pow(2).sum().sqrt()
            aug = self.directions[direction_index, :].mul(weights).sum(0)
        return x.detach(), y, aug.detach()

augmented_dataset = ImbanlanceAugmented(train_dataset, torch.tensor(top_directions).float())
augmented_loader = torch.utils.data.DataLoader(augmented_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           sampler=train_sampler)

classifier = torch.nn.Linear(256, 100).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), momentum=0.9, lr=1e-3, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(10):
    for batch_idx, (data, target, aug) in enumerate(augmented_loader):
        classifier.train()
        data, target, aug = data.to(device), target.to(device), aug.to(device)
        optimizer.zero_grad()
        features = model_get_features(model.module, data).detach()
        outputs = classifier(features + 1*aug*np.random.choice([-1, 1]))
        loss = criterion(outputs, target.to(torch.long))
        loss.backward()
        optimizer.step()
    
        if batch_idx % 100 == 10000:
            print(f"Train Epoch: {epoch} [{batch_idx}/{len(augmented_dataset)} "
                  f"({100. * batch_idx / len(augmented_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    
    #scheduler.step()
    
    
    model.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_total = 0
    correct_per_class = {d:0 for d in range(100)}
    total_per_class = {d:0 for d in range(100)}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.squeeze(0), target.squeeze(0)
            output = classifier(model_get_features(model.module, data))
            loss = criterion(output, target.to(torch.long))
            test_loss += loss.item()
            test_total += 1
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
            
            for d in range(100):
                correct_per_class[d] += pred.eq(target.view_as(pred)).flatten()[target==d].sum().item()
                total_per_class[d] += (target==d).sum().item()
            
    test_loss = test_loss / test_total
    test_accuracy = correct / total
    print(f"epoch {epoch} | Test accuracy: {test_accuracy} | Test loss: {test_loss:.4f}")

# Test model

#classifier = model.module.fc_cb

criterion = torch.nn.CrossEntropyLoss()

model.eval()
classifier.eval()
test_loss = 0
correct = 0
total = 0
test_total = 0
correct_per_class = {d:0 for d in range(100)}
total_per_class = {d:0 for d in range(100)}
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = data.squeeze(0), target.squeeze(0)
        output = classifier(model_get_features(model.module, data))
        loss = criterion(output, target.to(torch.long))
        test_loss += loss.item()
        test_total += 1
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(target)
        
        for d in range(100):
            correct_per_class[d] += pred.eq(target.view_as(pred)).flatten()[target==d].sum().item()
            total_per_class[d] += (target==d).sum().item()
        
test_loss = test_loss / test_total
test_accuracy = correct / total
print()
print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")
print()
for d in range(100):
    print(f"Accuracy for class {d}: {correct_per_class[d]/total_per_class[d]}")
print()
print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")