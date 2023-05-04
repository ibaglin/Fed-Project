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

transform_train2 = transforms.Compose([transforms.ToPILImage(), torchvision.transforms.Grayscale(num_output_channels=1),transforms.CenterCrop(300),transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


dataset = CustomImageDataset(annotations_file= 'dataset_file.csv', img_dir= 'images', transform= transform_train2)

train_loader2 = DataLoader(dataset, batch_size=1, shuffle=False)


data_test_list =[]
target_test_list =[]
for j, (data, target) in enumerate(train_loader2):
    data, target = data.cuda(), target.cuda()
    data_test_list.append(data)
    target_test_list.append(target)
  
   
model = torch.load('model_scripted.pt')

classes=['0','1','2','3','4','5','6','7','8','9']
for  (data, target) in zip(data_test_list,target_test_list):
    
    output = model(data)
    _,pred = output.max(1)
    
    print(classes[pred[0]])
    