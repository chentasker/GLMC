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
min_dist = np.inf
with torch.no_grad():
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        _, _, _, _, features = model.module(data, train=True, extract_features=True)
        
                #features = model_get_features(model.module, data)
        outputs = model.module.fc_cb(features)
        pi = (torch.linalg.pinv(model.module.fc_cb.weight.data) @ (outputs - model.module.fc_cb.bias.data).T).T
        kz = (features - pi).detach().cpu().numpy()
        
        kz_to_test = kz[1]
        coarse_to_test = coarse_labels[target[0]]
        image_to_test = data[0]
        target_to_test = target[0]
        pi_to_test = pi[0]
        pi_image = data[0]
        kz_image = data[1]
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        _, _, _, _, features = model.module(data, train=True, extract_features=True)
        
                #features = model_get_features(model.module, data)
        outputs = model.module.fc_cb(features)
        pi = (torch.linalg.pinv(model.module.fc_cb.weight.data) @ (outputs - model.module.fc_cb.bias.data).T).T
        kz = (features - pi).detach().cpu().numpy()
        
        dist = np.linalg.norm(features.cpu().numpy() - (pi_to_test.cpu().numpy()+kz_to_test),axis=1)
        #dist = dist[target.cpu() != target_to_test.cpu()]
        #data = data[target.cpu() != target_to_test.cpu()]
        #target = target[target.cpu() != target_to_test.cpu()]
        
        if np.min(dist) < min_dist:
            min_coarse = coarse_labels[target[np.argmin(dist)]]
            min_label = target[np.argmin(dist)]
            min_dist = np.min(dist)
            min_data = data[np.argmin(dist)]


#image_np = image_to_test.permute(1, 2, 0).cpu().numpy()
import torchvision.transforms as T
# If the tensor is normalized, unnormalize it:
# Assuming standard CIFAR-100 normalization
unnorm = T.Normalize(mean=[-0.5071/0.2675, -0.4867/0.2565, -0.4408/0.2761],
                     std=[1/0.2675, 1/0.2565, 1/0.2761])
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
for ax,im in zip(axes,[pi_image, kz_image, min_data]):
    image_unnorm = unnorm(im).permute(1, 2, 0).cpu().numpy()
    
    # Clip and show
    ax.imshow(image_unnorm.clip(0, 1))
    ax.axis('off')
plt.show()