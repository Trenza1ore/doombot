import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import sample, randrange
from collections import deque
from numpy.random import default_rng, randint
from numpy import expand_dims, ndarray, stack, arange, argmax, array, repeat, tile
import numpy as np

model_savepath = "pretrained/model-doom-%d.pth"

class DQNv1(nn.Module):
    def __init__(self, action_num: int, dropout: float=0):
        super().__init__()
        if dropout == 0:
            self.conv_layer_0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_1 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_2 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False),
                nn.ReLU()
            )
            
            self.hidden_layers = nn.Sequential(
                nn.Linear(6272, 1568, bias=True),
                nn.ReLU(),
                nn.Linear(1568, action_num, bias=True),
                nn.ReLU()
            )
            
        else:
            self.conv_layer_0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False),
                nn.Dropout2d(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_1 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False),
                nn.Dropout2d(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_2 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False),
                nn.Dropout2d(),
                nn.ReLU()
            )
            
            self.hidden_layers = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(6272, 1568, bias=True),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(1568, action_num, bias=True),
                nn.ReLU()
            )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = x.view(-1, 4608)
        return self.hidden_layers(x)
        