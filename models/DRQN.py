import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import sample, randrange
from collections import deque
from numpy.random import default_rng, randint
from numpy import expand_dims, ndarray, stack, arange, argmax, array, repeat, tile
import numpy as np
from models import DQNv1

model_savepath = "pretrained/model-doom-%d.pth"

class DRQNv1(nn.Module):
    def __init__(self, action_num: int, dropout: float=0):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, bias=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True)
        )
        
        self.decision_net = nn.Sequential(
            nn.LSTM(input_size=4608, hidden_size=action_num, bias=True, dropout=dropout)
        )
        
        self.feature_net = nn.Sequential(
            nn.Dropout(p=dropout)
            nn.Linear(4608, 512, bias=True),
            nn.Dropout(p=dropout)
            nn.Sigmoid(),
            nn.Linear(512, feature_num, bias=True),
            nn.Sigmoid()
        )
    
    def inf_feature(self, x: torch.Tensor):
        x = self.conv_layers(x)
        x = x.view(-1, 4608)
        self.x_copy = torch.clone(x)
        return self.feature_net(x)
    
    def inf_action(self):
        actions = self.decision_net(self.x_copy)[0]
        self.x_copy = None
        return actions
    
    def forward(self, x: torch.Tensor):
        feature = self.inf_feature(x)
        actions = self.inf_action()
        return (feature, actions)
        
class DRQNv2(DQNv1):
    def __init__(self, action_num: int, feature_num: int, dropout: float=0):
        super().__init__()
        
        self.decision_net = nn.Sequential(
            nn.LSTM(input_size=6272, hidden_size=action_num, bias=True, dropout=dropout)
        )
        
        if dropout == 0:
            self.feature_net = nn.Sequential(
                nn.Linear(6272, 1568, bias=True),
                nn.Sigmoid(),
                nn.Linear(1568, feature_num, bias=True),
                nn.Sigmoid()
            )
            
        else:
            self.feature_net = nn.Sequential(
                nn.Dropout(p=dropout)
                nn.Linear(6272, 1568, bias=True),
                nn.Dropout(p=dropout)
                nn.Sigmoid(),
                nn.Linear(1568, feature_num, bias=True),
                nn.Sigmoid()
            )
    
    def inf_feature(self, x: torch.Tensor):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = x.view(-1, 4608)
        self.x_copy = torch.clone(x)
        return self.feature_net(x)
    
    def inf_action(self):
        actions = self.decision_net(self.x_copy)[0]
        self.x_copy = None
        return actions
    
    def forward(self, x: torch.Tensor):
        feature = self.inf_feature(x)
        actions = self.inf_action()
        return (feature, actions)