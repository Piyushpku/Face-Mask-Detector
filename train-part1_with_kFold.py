# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 01:09:24 2021

@author: Piyush
"""
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
from torchvision.datasets import ImageFolder
from sklearn import metrics
tic = time.time()

num_epochs = 5
num_classes = 3
learning_rate = 0.0001
prec=[]
rec=[]
f1=[]
data_dir = './Dataset'
classes = os.listdir(data_dir + "/Train")
train_tfms= transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_tfms= transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_ds = ImageFolder(data_dir+'/Train', train_tfms)
test_ds = ImageFolder(data_dir+'/Test', test_tfms)
batch_size=8
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size, num_workers=4, pin_memory=True)
print(classes)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(128 * 8 * 8, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1000, 3)
           
        )

    def forward(self, x):

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

def main():
    
  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=10 , shuffle=True)
  model = CNN()

  print('--------------------------------')
  # K-fold Cross Validation model evaluation
  for fold,(train_ids, test_ids) in enumerate(kfold.split(train_loader.dataset)):
    print(f'FOLD {fold+1}')
    print('--------------------------------')   
    # Sample elements randomly from a given list of ids, no replacement.
    train_set = torch.utils.data.dataset.Subset(train_loader.dataset,train_ids)
    val_set = torch.utils.data.dataset.Subset(train_loader.dataset,test_ids)
    # Define data loaders for training and testing data in this fold
    trainloader = DataLoader(train_set, batch_size=8,shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)
    for epoch in range(0,num_epochs):
        print(f'Starting epoch {epoch+1}')
        train(model,trainloader,epoch)     
    evaluation(model,fold+1,val_loader) 
  show_average()
  evaluation(model,0,test_loader) 
  torch.save(model, './Model/old.pth')
  plt.legend(frameon=False)
  plt.show()

def show_average():
    Precision=0.0
    Recall=0.0
    fs=0.0
    for i in range(0,10):
        Precision+=prec[i]
        Recall+=rec[i]
        fs+=f1[i]
    Precision=(Precision/10.0)
    Recall=(Recall/10.0)
    fs=(fs/10.0)
    print("\n******Average Statistics for 10-folds********")
    print("\nPrecision : {}".format(Precision))
    print("\nRecall : {:.4f}".format(Recall))
    print("\nf1-score : {:.4f}\n".format(fs))
    
    
def train(model,trainloader,epoch):
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  total_step = len(trainloader)
  loss_list = []
  acc_list = []
  for i,data in enumerate(trainloader):
            # Forward pass
            images,labels=data
            optimizer.zero_grad()  
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss_list.append(loss.item())
    
            # Backprop and perform Adam optimisation
           # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
    
            if (i + 1) % 50== 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
  plt.plot(loss_list, label='Training loss')
    
def evaluation(model,fold,testloader):  
  model.eval()
  y_pred=[]
  y_act=[]
  with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i in range(0,len(labels)):
                y_act.append(classes[labels[i]])
                y_pred.append(classes[predicted[i]])
            correct += (predicted == labels).sum().item()
       
        if fold==0:
            print('*****Test Accuracy of the model on 600 test images: {} %'.format((correct / total) * 100))
        else:
            precision,recall,fscore,support=metrics.precision_recall_fscore_support(y_act,y_pred,average='weighted')
            prec.append(precision)
            rec.append(recall)
            f1.append(fscore)
            print(f'*****Test Accuracy of the model in fold {fold} '+': {} %'.format((correct / total) * 100))
        print('*****Confusion Matrix')
        print(metrics.confusion_matrix(y_act,y_pred))
        cm = metrics.confusion_matrix(y_act,y_pred)
        cmd = metrics.ConfusionMatrixDisplay(cm, display_labels=['Mask','NonMask','Others'])
        cmd.plot()
        if fold==0:
            print('*****Classification Report')
            print(metrics.classification_report(y_act,y_pred))
            toc = time.time()
            print('duration = ', toc - tic)

if __name__ == '__main__':
    main()