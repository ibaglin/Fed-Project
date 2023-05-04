import os
import random
import numpy as np
import torch, torchvision

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import vgg19
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torchvision.io import read_image
import pandas as pd
from skimage import io
from torch.autograd import Variable

num_clients = 5
rounds = 100
epochs = 5

stddev = 0.2
state = 0

std=5
poor_data=0

scale_reduction = 1

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        label =torch.tensor(int(self.img_labels.iloc[idx,1]))
       

        if self.transform:
            image = self.transform(image)
        return image, label


############## Load the Dataset  ############## 


transform_train1 = transforms.Compose([transforms.ToTensor(),torchvision.transforms.Grayscale(num_output_channels=1),transforms.CenterCrop(300), transforms.Normalize((0.5,), (0.5,))])

transform_train2 = transforms.Compose([transforms.ToPILImage(), torchvision.transforms.Grayscale(num_output_channels=1),transforms.CenterCrop(300),transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

if poor_data==1:
   transform_train = transform_train1
else:
   transform_train = transform_train2

dataset = CustomImageDataset(annotations_file= 'dataset_file.csv', img_dir= 'images', transform= transform_train2)
test_set = CustomImageDataset(annotations_file= 'test_file.csv', img_dir= 'images2', transform= transform_train2)


node1,node2,node3,node4,node5 = random_split(dataset, [80,80,80,80,79])
train_split=[]
train_split.append(node1)
train_split.append(node2)
train_split.append(node3)
train_split.append(node4)
train_split.append(node5)
train_loader = [DataLoader(x, batch_size=32, shuffle=True) for x in train_split]

train_loader2 = DataLoader(dataset, batch_size=32, shuffle=False)

test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
print(len(train_loader))
transform_test = torch.nn.Sequential(transforms.GaussianBlur(kernel_size=(5, 9), sigma=(std, std)) )

   
############## Storing Data on the devices (not to be moved later)  ############## 
data_outer_list=[]
target_outer_list=[]

for i in range(num_clients):
    data_inner_list=[]
    target_inner_list=[]
    for j, (data, target) in enumerate(train_loader[i]):
        data, target = data.cuda(), target.cuda()
        data_inner_list.append(data)
        
        target_inner_list.append(target)
    data_outer_list.append(data_inner_list)
    target_outer_list.append(target_inner_list)
    

       
############## Convolutional Neural Network  ############## 
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.convolutaional_neural_network_layers = nn.Sequential(

                nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1), # (N, 1, 28, 28) 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), 
                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
              
                nn.MaxPool2d(kernel_size=2) 
             
        )
        self.linear_layers = nn.Sequential(
              
                nn.Linear(in_features=135000, out_features=64),          
                nn.ReLU(),
                nn.Dropout(p=0.2), 
                nn.Linear(in_features=64, out_features=10) 
        )
   
    def forward(self, x):
        x = self.convolutaional_neural_network_layers(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.linear_layers(x)
        return x
              
############## This function updates/trains node model on node data  ############## 
def node_update(node_model, optimizer,data_list,target_list, device_num,epoch):

    model.train()
    
    for i in range(epoch):
        for  (data, target) in zip(data_list,target_list):
            optimizer.zero_grad()
            loss_fn = nn.CrossEntropyLoss()
            output = node_model(data)
            
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
    

############## This function will aggregate the mean of the parameters of the model ##############  
def aggregate(global_model,model_list,stddev,test_loader,state):

    state_dict_list= []
    for model in model_list:
        state_dict_list.append(model.state_dict())

    
    global_dict = global_model.state_dict()
    new_global_dict = global_dict
    for layer in global_dict:
       new_global_dict[layer] = torch.zeros_like(global_dict[layer])
       for state_dict in state_dict_list:
           new_global_dict[layer] += state_dict[layer]
       global_dict[layer] = new_global_dict[layer] / len(state_dict_list)
          
        
    global_model.load_state_dict(global_dict)
    
    for i,model in enumerate(model_list):
        model.load_state_dict(global_model.state_dict())
        
        
    return model_list

############## This function test the global model on test data and returns test loss and test accuracy ##############        
def global_test( global_model, test_loader):
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)   
            correct += (output.max(dim=1)[1] == target).sum()
    acc = correct / len(test_loader.dataset)
    return  acc
    
      
############## client models ##############
global_model =  Net().cuda(0)
model_list =[]

for i in range(num_clients):
   model_list.append( Net().cuda())             


############## synchronizing with global model ##############
for model in model_list:
    model.load_state_dict(global_model.state_dict()) 

opt = [optim.SGD(model.parameters(), lr=0.1) for model in model_list]

############## Iterate through rounds of training ##############
for r in range(rounds):

    ######loss = 0
    for i in range(num_clients):
       node_update(model_list[i], opt[i],data_outer_list[i],target_outer_list[i],i, epochs) 
        
    
    model_list = aggregate(global_model,model_list,stddev,test_loader,state)
    acc = global_test( global_model, test_loader)
    ####print('Round %d' % r)
    print(' %0.3f' %  acc)
   
##model_scripted = torch.jit.script(global_model)
##model_scripted.save('model_scripted.pt')


    