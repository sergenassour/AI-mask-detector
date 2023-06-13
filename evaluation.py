import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#Data Transform

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#Data Setup

test_path = 'testing_data'
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size = 50, shuffle = True
)

class Evaluation:

   def __init__(self, y_true, y_pred):
    self.y_true = y_true
    self.y_pred = y_pred


    def confusionMatrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred, average='micro')

    def recall(self):
        return recall_score(self.y_true, self.y_pred, average='micro')

    def f1(self):
        return f1_score(self.y_true, self.y_pred, average='micro')
