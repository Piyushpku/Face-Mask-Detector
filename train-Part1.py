# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:09:05 2021

@author: Piyush
"""
import matplotlib.pyplot as plt
import os
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
    model = CNN()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
    
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
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
    torch.save(model, './Model/mask_detection-cnn.pth')
    plt.plot(loss_list, label='Training loss')
    plt.legend(frameon=False)
    plt.show()
    model.eval()
    y_pred=[]
    y_act=[]
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i in range(0,len(labels)):
                y_act.append(classes[labels[i]])
                y_pred.append(classes[predicted[i]])
            correct += (predicted == labels).sum().item()
        print('*****Test Accuracy of the model on 600 test images: {} %'.format((correct / total) * 100))
        label=['Mask', 'NonMask', 'NotPerson']
        print('*****Confusion Matrix')
        print(metrics.confusion_matrix(y_act,y_pred,labels=label))
        print('*****Classification Report')
        print(metrics.classification_report(y_act,y_pred))
    toc = time.time()
    print('duration = ', toc - tic)

if __name__ == '__main__':
    main()