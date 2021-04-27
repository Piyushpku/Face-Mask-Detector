# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:57:14 2021

@author: Piyush
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:09:05 2021

@author: Piyush
"""
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import time
from torchvision.datasets import ImageFolder
from sklearn import metrics
import pandas as pd
tic = time.time()
classes = ['Mask', 'NonMask', 'NotPerson']
accuracy=[]

batch_size=8
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(     
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(128 * 7 * 7, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(500, 3)
           
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
    model = torch.load('./Model/latest.pth')
    #evaluation(model,'./Dataset/Test','CONFUSION MATRIX')
    data_dir = './Dataset/Bias'
    print('*********Bias analysis on Gender*********')
    print('\n------Data for Males-------\n')
    path=data_dir+'/Gender/Male'
    evaluation(model,path,'CONFUSION MATRIX FOR MALES')
    print('\n------Data for Females------\n')
    path=data_dir+'/Gender/Female'
    evaluation(model,path,'CONFUSION MATRIX FOR FEMALES')
    
    print('\n\n*********Bias analysis on Race*********')
    print('Data for Asians')
    path=data_dir+'/Race/Asian'
    evaluation(model,path,'CONFUSION MATRIX FOR ASIANS')
    print('Data for Blacks')
    path=data_dir+'/Race/Black'
    evaluation(model,path,'CONFUSION MATRIX FOR BLACKS')
    print('Data for Whites')
    path=data_dir+'/Race/White'
    evaluation(model,path,'CONFUSION MATRIX FOR WHITES')
    print('\n\n\n*************************BIAS ANALYSIS SUMMARY USING F-1 SCORE*************************\n')
    print('GENDER      MALE        FEMALE')
    print('            {}     {}'.format(accuracy[0],accuracy[1]))
    print('\nRACE        ASIAN       BLACK        WHITE')
    print('            {}     {}      {}'.format(accuracy[2],accuracy[3],accuracy[4]))
   # for acc in accuracy:
       
 
def evaluation(model,path,title):   
    test_tfms= transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = ImageFolder(path, test_tfms)
    test_loader = DataLoader(test_ds, batch_size, num_workers=4, pin_memory=True)
    model.eval()
    y_act=[]
    y_pred=[]
    with torch.no_grad():
        correct = 0
        total = 0
        count=1
        c=1;
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(0,len(labels)):
                y_act.append(classes[labels[i]])
                y_pred.append(classes[predicted[i]])
                #to_pil = transforms.ToPILImage()
                #fig=plt.figure(figsize=(30,30))
                #image = to_pil(images[i])
                index = predicted[i]
                #sub = fig.add_subplot(1, 19,count)
                if labels[i] == predicted[i]:
                    res=True
                else:
                    res=False
                #sub.set_title(str(classes[index]) + ":" + str(res))
                #plt.axis('off')
                count=count+1
                #plt.imshow(image)
                #figname = 'fig_{}.png'.format(c)
                c=c+1
                #plt.imshow(image)
                #dest = os.path.join('./Output', figname)
                #plt.savefig(dest)  # write image to file  
                #plt.cla()
                if count==19:
                    plt.show()
                    count=1 
                    plt.close('all')
            #plt.close('all')
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc=float((correct / total) * 100)
        print('*****Test Accuracy of the model on the sample images: {} %'.format(acc))
        label=['Mask', 'NonMask', 'NotPerson']
        print('*****Confusion Matrix')
        cm = metrics.confusion_matrix(y_act,y_pred,labels=label)
        cmtx = pd.DataFrame(
            metrics.confusion_matrix(y_act, y_pred, labels=['Mask', 'NonMask','NotPerson']), 
            index=['true:Mask', 'true:NoMask','true:NotPerson'], 
            columns=['pred:Mask', 'pred:NoMask','ped:NotPerson']
        )  
        print(cmtx)
        print()
        cmd = metrics.ConfusionMatrixDisplay(cm, display_labels=['Mask','NonMask','Others'])
        cmd.plot()
        cmd.ax_.set(xlabel=title,ylabel='')
        print('*****Classification Report')
        print(metrics.classification_report(y_act,y_pred,labels=label))
       # macro_f1 = report['macro avg']['f1-score']/100
        precision,recall,fscore,support=metrics.precision_recall_fscore_support(y_act,y_pred,average='weighted')
        macro_f1= "{:.4f}".format(fscore)
        print(macro_f1)
        accuracy.append(macro_f1)
        plt.show()
    toc = time.time()
    print('duration = ', toc - tic)

if __name__ == '__main__':
    main()