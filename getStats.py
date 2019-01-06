import torch
from data import TestDataset
import train_config as config
import numpy as np

train_img_list = np.load(config.train['train_img_list']).tolist()
val_img_list = np.load(config.train['val_img_list']).tolist()

train_dataset = TestDataset( train_img_list, path = '../../gopro_dataset/')
val_dataset = TestDataset( val_img_list, path = '../../gopro_dataset/')


tmp_mean = torch.zeros(3, dtype=torch.float32)
tmp_std = torch.zeros(3, dtype=torch.float32)

for img in train_dataset:
    tmp_mean += img['img256'].sum(-1).sum(-1)

for img in val_dataset:
    tmp_mean += img['img256'].sum(-1).sum(-1)

mean = tmp_mean / 3214 /720/1280
print(mean)

for img in train_dataset:
    for i in range(3):
        tmp_std[i] += ((img['img256'][i,:,:] - mean[i])**2).sum(-1).sum(-1)

for img in val_dataset:
    for i in range(3):
        tmp_std[i] += ((img['img256'][i,:,:] - mean[i])**2).sum(-1).sum(-1)

std = torch.sqrt(tmp_std / 3214/720/1280)
print(std)