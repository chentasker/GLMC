#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:14:32 2025

@author: chent
"""
min_coarse = 1
min_index = 0
kz_to_test = kz_tensor_for_coarse[0][0,:]
min_dist = np.linalg.norm(kz_tensor_for_coarse[1][0,:] - kz_to_test)
for c, kz_tensor in kz_tensor_for_coarse.items():
    dist = np.linalg.norm(kz_tensor - kz_to_test,axis=1)
    if np.min(dist) < min_dist and np.min(dist) > 0:
        min_coarse = c
        min_index = np.argmin(dist)
        min_dist = np.min(dist)

#%%
import matplotlib.pyplot as plt
import torchvision.transforms as T

if dataset == 'cifar10':
    train_dataset = cifar10Imbanlance.Cifar10Imbanlance(transform=transform_val,
                                                        imbanlance_rate=args.imbanlance_rate,
                                                        train=True,
                                                        file_path=args.root,
                                                        print_on_creation=False
                                                        )
    
if dataset == 'cifar100':
    train_dataset = cifar100Imbanlance.Cifar100Imbanlance(transform=transform_val,
                                                          imbanlance_rate=args.imbanlance_rate,
                                                          train=True,
                                                          file_path=os.path.join('data/','cifar-100-python/'),
                                                          print_on_creation=False
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
min_dist = np.inf

unnorm = T.Normalize(mean=[-0.5071/0.2675, -0.4867/0.2565, -0.4408/0.2761],
                     std=[1/0.2675, 1/0.2565, 1/0.2761])


with torch.no_grad():
    kz_index = torch.randint(low=0, high=len(train_dataset), size=(1,)).item()
    pi_index = torch.randint(low=0, high=len(train_dataset), size=(1,)).item()
    kz_data, kz_target = train_dataset[kz_index]
    kz_data = kz_data.unsqueeze(0).to(device)
    _, _, _, _, kz_features = model.module(kz_data, train=True, extract_features=True)
    outputs = model.module.fc_cb(kz_features)
    kz_to_test = (kz_features - (torch.linalg.pinv(model.module.fc_cb.weight.data) @ (outputs - model.module.fc_cb.bias.data).T).T)
    kz_coarse = coarse_labels[kz_target]
    pi_data, pi_target = train_dataset[pi_index]
    pi_data = pi_data.unsqueeze(0).to(device)
    _, _, _, _, pi_features = model.module(pi_data, train=True, extract_features=True)
    outputs = model.module.fc_cb(pi_features)
    pi_to_test = (torch.linalg.pinv(model.module.fc_cb.weight.data) @ (outputs - model.module.fc_cb.bias.data).T).T
    pi_coarse = coarse_labels[pi_target]
         
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        _, _, _, _, features = model.module(data, train=True, extract_features=True)
        
                #features = model_get_features(model.module, data)
        outputs = model.module.fc_cb(features)
        pi = (torch.linalg.pinv(model.module.fc_cb.weight.data) @ (outputs - model.module.fc_cb.bias.data).T).T
        kz = features - pi
        
        pi_dist = torch.linalg.norm(pi - pi_to_test,axis=1).detach().cpu().numpy()
        pi_dist = pi_dist / np.linalg.norm(pi_dist)
        kz_dist = torch.linalg.norm(kz - kz_to_test,axis=1).detach().cpu().numpy()
        kz_dist = kz_dist / np.linalg.norm(kz_dist)
        dist = np.sqrt(kz_dist**2 + pi_dist**2)
        #dist = dist[target.cpu() != target_to_test.cpu()]
        #data = data[target.cpu() != target_to_test.cpu()]
        #target = target[target.cpu() != target_to_test.cpu()]
        
        if np.min(dist) < min_dist:
            min_coarse = coarse_labels[target[np.argmin(dist)]]
            min_label = target[np.argmin(dist)]
            min_dist = np.min(dist)
            min_data = data[np.argmin(dist)]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
for ax,im in zip(axes,[pi_data, kz_data, min_data]):
    image_unnorm = unnorm(im.squeeze()).permute(1, 2, 0).cpu().numpy()
    ax.imshow(image_unnorm.clip(0, 1))
    ax.axis('off')
plt.show()