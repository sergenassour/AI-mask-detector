# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:33:42 2022

@author: stefa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import os

#training data and test data setup
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#Data Setup
dataset_path = 'dataset'

complete_dataset_loader = DataLoader(
    torchvision.datasets.ImageFolder(dataset_path, transform=transformer),
    batch_size = 32, shuffle = True
)

train_size = int(0.8 * len(complete_dataset_loader))
test_size = len(complete_dataset_loader) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(complete_dataset_loader, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

#CNN architecture
classes = ("cloth", "n95", "none", "surgical")

learningRate = 0.01

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
    #these are the layers of our network, to be modified  
        self.convolutionLayer1 = nn.Conv2d(3, 16, 3, 1, 1) #change input size and output size, kernel size = 3x3, stride = 1, padding = 1
        self.convolutionLayer2 = nn.Conv2d(16, 32, 3, 1, 1) #change input size and output size, kernel size = 3x3, stride = 1, padding = 1
        self.convolutionLayer3 = nn.Conv2d(32, 32, 3, 1, 1) #change input size and output size, kernel size = 3x3, stride = 1, padding = 1
        self.poolingLayer = nn.MaxPool2d(2, 2)
        self.fullyConnectedLayer1 = nn.Linear(32768, 1024)
        self.fullyConnectedLayer2 = nn.Linear(1024, 512)
        self.fullyConnectedLayer3 = nn.Linear(512, 4)
    #this is the forward pass
    def forward(self, output):
        
        output = self.poolingLayer(F.relu(self.convolutionLayer1(output))) 
        output = self.poolingLayer(F.relu(self.convolutionLayer2(output)))
        output = self.poolingLayer(F.relu(self.convolutionLayer3(output))) 
        
        output = output.view(output.size(0), -1) #flatten tensor
        
        output = F.relu(self.fullyConnectedLayer1(output))
        output = F.relu(self.fullyConnectedLayer2(output))
        output = self.fullyConnectedLayer3(output)
        
        return output

model = NeuralNet() #this is our model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

totalStep = len(train_loader)
lossList = []
accurateList = []
numEpochs = 10

#training loop
for epoch in range(numEpochs):
    for i, (images, labels) in enumerate(train_loader):
        
        #forward pass
        out = model(images)
        loss = criterion(out, labels)
        lossList.append(loss.item())
        
        #backpropagation and optimization
        optimizer.zero_grad() #set gradient to zero
        loss.backward()
        optimizer.step()
        
        #accuracy
        
        total = labels.size(0)
        _, predicted = torch.max(out.data, 1)
        correct = (predicted == labels).sum().item()
        accurateList.append(correct / total)

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, numEpochs, i + 1, totalStep, loss.item(),
                          (correct / total) * 100))

print("Training Finished")
model.eval()


y_pred = []
y_true = []

# iterate over test data
for images, labels in test_loader:
        output = model(images) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# constant for classes
classes = ("cloth", "n95", "none", "surgical")

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')

print("Precision: ", precision_score(y_true, y_pred, average = 'micro'))
print("Recall: ", recall_score(y_true, y_pred,average = 'micro'))
print("F1-measure: ", f1_score(y_true, y_pred,average = 'micro '))
print("Accuracy: ", accuracy_score(y_true, y_pred))

print("Confusion Matrix: ")
print(classification_report(y_true, y_pred, target_names = classes))

FILE = "model.pth"
torch.save(model.state_dict(), FILE) #save model

#load model
load_model = model()
load_model.load_state_dict(torch.load(FILE))
load_model.eval()

