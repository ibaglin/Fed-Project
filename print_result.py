import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
def  show_images ( images ): 
    for  index ,  image  in  enumerate ( images ): 
        plt.figure(index)
        plt.axis('off')
        plt . imshow ( image . reshape ( 28 ,  28 ))
        name = 'test_images' + str(index) + '.png'
        plt.savefig(name)
        plt.close()       

    
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

G = generator()
G = torch.load('Generator_epoch_200.pth')
G.eval()
G.cpu()
noise = (torch.rand(400, 128)-0.5) / 0.5
noise.cpu()
fake_image = G(noise).cpu()

imgs_numpy  =  ( fake_image . data . cpu () . numpy () + 1.0 ) / 2.0 
show_images ( imgs_numpy ) 



