import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import sample, randrange
from collections import deque
from numpy.random import default_rng, randint
from numpy import expand_dims, ndarray, stack, arange, argmax, array, repeat, tile

model_savepath = "pretrained/model-doom-%d.pth"

class DQNv2(nn.Module):
    def __init__(self, action_num: int, batch_size: int, device):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.AvgPool2d(4, 4)
            )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        # self.lstm1 = nn.Sequential(
        #     nn.LSTM(15984, )
        # )
        self.fc1 = nn.Sequential(
            nn.Linear(15984, 11988, bias=False),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(11988, 8991, bias=False),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(8991, action_num, bias=False),
            nn.ReLU()
        )
        self.sobel_x = torch.tensor([
            [ 1,  0, -1],
            [ 2,  0, -2],
            [ 1,  0, -1]
        ]).float().reshape(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([
            [ 1,  2,  1],
            [ 0,  0,  0],
            [-1, -2, -1]
        ]).float().reshape(1, 1, 3, 3).to(device)
        self.sobel_xb = self.sobel_x.repeat(batch_size, 1, 1, 1).to(device)
        self.sobel_yb = self.sobel_y.repeat(batch_size, 1, 1, 1).to(device)
        self.batch_size = batch_size
    
    def sobel_batch(self, x: torch.tensor):
        grayscale = x[:, 0, :, :]
        dx = F.conv2d(grayscale, self.sobel_xb, padding=1, groups=self.batch_size)
        dy = F.conv2d(grayscale, self.sobel_yb, padding=1, groups=self.batch_size)
        return torch.sqrt(dx*dx+dy*dy).unsqueeze(1)
    
    def sobel_single(self, x: torch.tensor):
        grayscale = x[:, 0, :, :]
        dx = F.conv2d(grayscale, self.sobel_x, padding=1, groups=1)
        dy = F.conv2d(grayscale, self.sobel_y, padding=1, groups=1)
        return torch.sqrt(dx*dx+dy*dy).unsqueeze(1)
        
    def forward(self, x: torch.tensor):
        #print(x.shape)
        sobel = self.sobel_batch(x)
        #print(sobel.shape)
        x = torch.cat([x, sobel], dim=1)
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #print(x.shape)
        x = x.view(-1, 15984)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def inference(self, x: torch.tensor):
        sobel = self.sobel_single(x)
        x = torch.cat([x, sobel], dim=1)
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 15984)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x