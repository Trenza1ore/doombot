import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import sample, randrange
from collections import deque
from numpy.random import default_rng, randint
from numpy import expand_dims, ndarray, stack, arange, argmax, array, repeat, tile

from models.legacy.dqn_agent import DQN_agent
from models.legacy.network import DQNv2, model_savepath

class Double_DQN_agent(DQN_agent):
    def __init__(self, device: torch.device, action_num: int, mem_size: int, 
                 batch_size: int, discount: float, lr: float, wd: float,
                 replay_mem_runs: int=10, model=DQNv2,
                 eps: float=1, eps_decay: float=0.99, eps_min: float=0.1):
        
        super().__init__(device, action_num, mem_size, batch_size, discount, lr, 
                         wd, replay_mem_runs, model, eps, eps_decay, eps_min)
        self.double_net = model(action_num, batch_size, device).to(device)
        self.double_net.load_state_dict(self.q_net.state_dict())
    
    def update_double_net(self):
        self.double_net.load_state_dict(self.q_net.state_dict())
    
    def train(self):
        #batch = sample_mem(self.memory, self.replay_mem_runs, self.rand_num)
        batch = sample(self.memory, self.batch_size)
        batch = array(batch, dtype=object)
        
        #print(batch[:, 0].shape, batch[:, 0].dtype)
        #print(batch[:, 0][0][0].shape, batch[:, 0][0][1].shape)
        device = self.device
        states = stack(batch[:, 0], dtype=float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = stack(batch[:, 3], dtype=float)
        terminated = batch[:, 4].astype(bool)
        on_going = ~terminated
        
        row_idx = self.batch_row_idx
        
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(device)
            net1_result = self.q_net(next_states).cpu().data.numpy()
            idx = (row_idx, argmax(net1_result, 1))
            next_state_values = self.double_net(next_states).cpu().data.numpy()[idx][on_going]
            
        q_targets = rewards.copy() # don't modify in-place and change memory
        q_targets[on_going] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(device)
        
        idx = (row_idx, actions)
        states = torch.from_numpy(states).float().to(device)
        action_values = self.q_net(states)[idx].float().to(device)
        
        self.optimiser.zero_grad()
        temporal_err = self.criterion(q_targets, action_values)
        temporal_err.backward()
        self.optimiser.step()
        
        self.eps = self.eps * self.eps_decay if self.eps > self.eps_min else self.eps_min