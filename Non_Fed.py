import os
import random
import numpy as np
import torch, torchvision


import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import vgg19
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import torch.utils.data as data_utils


rounds = 30
epochs = 5

stddev = 0.2
state = 0

std=5
poor_data=0

scale_reduction = 1 
imagelabel = 50000



############## Load the Dataset  ############## 

transform_train1 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),transforms.GaussianBlur(kernel_size=(5, 9), sigma=(std, std))])

transform_train2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

if poor_data==1:
   transform_train = transform_train1
else:
   transform_train = transform_train2
   
traindata = CIFAR10('./data', train=True, download=True, transform= transform_train)

split = torch.arange(imagelabel)
traindata = data_utils.Subset(traindata, split)

train_loader = DataLoader(traindata, batch_size=32, shuffle=True) 

test_loader = DataLoader(CIFAR10('./data', train=False, transform=transform_train), batch_size=32, shuffle=True)

transform_test = torch.nn.Sequential(transforms.GaussianBlur(kernel_size=(5, 9), sigma=(std, std)) )

############## Storing Data on the devices (not to be moved later)  ############## 
data_inner_list=[]
target_inner_list=[]

for j, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()
    data_inner_list.append(data)
    target_inner_list.append(target)
    

    
############## Convolutional Neural Network  ############## 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True)
                                    ,nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2, stride=2) 
                                    ,nn.Conv2d(64, 128, kernel_size=3, padding=1),nn.BatchNorm2d(128), nn.ReLU(inplace=True)
                                    ,nn.Conv2d(128, 128, kernel_size=3, padding=1),nn.BatchNorm2d(128), nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2, stride=2)
                                    ,nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
                                    ,nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),nn.ReLU(inplace=True)
                                    ,nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),nn.ReLU(inplace=True)
                                    ,nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)
                                    ,nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True)
                                    ,nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True) 
                                    ,nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True) 
                                    ,nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)
                                    ,nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True)
                                    ,nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True) 
                                    ,nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True) 
                                    ,nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)
                                    ,nn.AvgPool2d(kernel_size=1, stride=1))

        self.classifier = nn.Sequential( nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 10))
        
       
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

############ Adding Noise to the parameters within a model  ############## 
def add_noise(model,s,state):
 if state == 1:
   with torch.no_grad():
       for parameter in model.parameters():
          noise = torch.randn(parameter.size()) * s  + 0
          noise = noise.to(parameter.get_device())
          parameter.add_(noise)
           

############## This function updates/trains node model on node data  ############## 
def node_update(global_model, optimizer,data_list,target_list,state,stddev,epoch):

    global_model.train()
    
    for i in range(epoch):
        for  (data, target) in zip(data_list,target_list):
            optimizer.zero_grad()
            output = global_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    
############## This function test the global model on test data and returns test loss and test accuracy ##############        
def global_test( global_model, test_loader):
    
    global_model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)   
            correct += (output.max(dim=1)[1] == target).sum()
    acc = correct / len(test_loader.dataset)
    return  acc
    
      
############## client models ##############
global_model =  CNN().cuda()
           
opt = optim.SGD(global_model.parameters(), lr=0.1) 

############## Iterate through rounds of training ##############
for r in range(rounds):
    add_noise(global_model,stddev,state)
    node_update(global_model, opt,data_inner_list,target_inner_list,state,stddev, epochs) 
    
    acc = global_test( global_model, test_loader)
    ####print('Round %d' % r)
    print('%0.3f' %  acc)