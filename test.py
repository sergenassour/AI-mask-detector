# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:11:01 2022

@author: stefa
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

#training data and test data setup
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#Data Setup
train_path = 'training_data'
test_path = 'testing_data'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size = 50, shuffle = True
)

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size = 50, shuffle = True
)

#CNN architecture
classes = ("none", "surgical", "n95", "cloth")


dataiter = iter(train_loader)
images, labels = dataiter.next()

convolutionLayer1 = nn.Conv2d(3, 16, 3, 1, 1)

poolingLayer = nn.MaxPool2d(2, 2)
convolutionLayer2 = nn.Conv2d(16, 32, 3, 1, 1)
poolingLayer = nn.MaxPool2d(2, 2)
convolutionLayer3 = nn.Conv2d(32, 32, 3, 1, 1)
fullyConnectedLayer1 = nn.Linear(32 * 32 * 32, 1024)

x = convolutionLayer1(images)
print(images.shape)
x = poolingLayer(x)
print(x.shape)
x = convolutionLayer2(x)
print(x.shape)
x = poolingLayer(x)
print(x.shape)
x = convolutionLayer3(x)
print(x.shape)
x = poolingLayer(x)
print(x.shape)
x = x.view(50, 32 * 32 * 32)
print("1", x.shape)
x = fullyConnectedLayer1(x)
print("2", x.shape)