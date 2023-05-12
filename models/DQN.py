import torch
import torch.nn as nn

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
                nn.Dropout(p=dropout),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_1 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False),
                nn.Dropout(p=dropout),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            
            self.conv_layer_2 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False),
                nn.Dropout(p=dropout),
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
        x = x.view(-1, 6272)
        return self.hidden_layers(x)

class DRQNv1(nn.Module):
    def __init__(self, action_num: int, feature_num: int):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, bias=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True)
        )
        
        self.decision_net = nn.Sequential(
            nn.LSTM(input_size=4608, hidden_size=action_num, bias=True)
        )
        
        self.feature_net = nn.Sequential(
            nn.Linear(4608, 512, bias=True),
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
        super().__init__(action_num, dropout)
        
        self.decision_net = nn.Sequential(
            nn.LSTM(input_size=6272, hidden_size=action_num, num_layers = 2, bias=True, dropout=dropout)
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
                nn.Dropout(p=dropout),
                nn.Linear(6272, 1568, bias=True),
                nn.Dropout(p=dropout),
                nn.Sigmoid(),
                nn.Linear(1568, feature_num, bias=True),
                nn.Sigmoid()
            )
    
    def inf_feature(self, x: torch.Tensor):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = x.view(-1, 6272)
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