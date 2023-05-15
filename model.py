import torch.nn as nn
from imports import *

transform1 = transforms.Compose([
    #transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(0.95)
])

def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        transform1,
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), 
        #batch normalization implementation
        nn.LazyBatchNorm2d(),
        nn.ReLU())

class model(nn.Module):
    def __init__(self, lr=0.01, num_classes=4):
        super().__init__()
        #self.save_hyperparameters()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=5, strides=3, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=3, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            #nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())

        
    def forward(self, x):
        return self.net(x)

def init_weights(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)