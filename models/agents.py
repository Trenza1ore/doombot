import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy.random import default_rng
import numpy as np

from models.DRQN import DRQNv1
from models.replay_memory import ReplayMemory
from vizdoom_utils import *
from time import time

model_savepath = "pretrained/nav-act-model-doom-%d.pth"

class CombatAgent:
    def __init__(self, device: torch.device, mem_size: int, 
                 action_num: int, nav_action_num: int,
                 discount: float, lr: float, loss=nn.MSELoss, 
                 act_wd: float=0, nav_wd: float=0, optimizer=optim.SGD, 
                 state_len: int=10, act_model=DRQNv1, nav_model=DRQNv1,
                 eps: float=1, eps_decay: float=0.99, eps_min: float=0.1,
                 seed=time()) -> None:
        
        # store model hyper parameters
        self.device = device
        self.action_num = action_num
        self.nav_action_num = nav_action_num
        self.discount = discount
        self.lr = lr
        self.act_wd = act_wd
        self.nav_wd = nav_wd
        self.optimiser = optimizer
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
        # set up random number generator
        self.rng = default_rng(seed)
        
        self.history_len = state_len - 2
        self.padding_len = state_len >> 1
        self.current_idx = self.history_len
        
        # set up memory for priorotized experience replay
        self.memory = ReplayMemory(res=(60, 108), ch_num=3, size=mem_size, 
                                   history_len=state_len-2, future_len=1, 
                                   dtypes=[np.uint8, np.float16, 'bool', 'bool', 'bool'])
        print(self.memory.features.dtype, self.memory.features.shape)
        
        # set up models
        self.criterion = loss()
        self.act_net = act_model(action_num=action_num, feature_num=1).to(device)
        self.nav_net = nav_model(action_num=nav_action_num, feature_num=1).to(device)
        self.act_optimizer = optimizer(self.act_net.parameters(), 
                                       lr=lr, weight_decay=act_wd)
        self.nav_optimizer = optimizer(self.nav_net.parameters(), 
                                       lr=lr, weight_decay=nav_wd)
    
    def decide_move_nav(self, state: np.ndarray) -> torch.tensor:
        if self.rng.uniform() < self.eps:
            return self.rng.integers(1, self.nav_action_num)
        else:
            state = torch.from_numpy(state).float().cuda()
            self.nav_net.inf_feature(state)
            return torch.argmax(self.nav_net.inf_action()).item()
    
    def decide_move(self, state: np.ndarray, is_combat: bool) -> torch.tensor:
        if self.rng.uniform() < self.eps:
            return self.rng.integers(1, self.action_num) if is_combat else self.rng.integers(1, self.nav_action_num)
        else:
            state = torch.from_numpy(state).float().cuda()
            if is_combat:
                self.act_net.inf_feature(state)
                return torch.argmax(self.act_net.inf_action()).item()
            else:
                self.nav_net.inf_feature(state)
                return torch.argmax(self.nav_net.inf_action()).item()
    
    def decide_move_blind(self, state: np.ndarray) -> torch.tensor:
        if self.rng.uniform() < self.eps:
            return self.rng.integers(1, self.action_num)
        else:
            state = torch.from_numpy(state).float().cuda()
            if self.act_net.inf_feature(state) > 0.5:
                return torch.argmax(self.act_net.inf_action()).item()
            else:
                self.nav_net.inf_feature(state)
                return torch.argmax(self.nav_net.inf_action()).item()
    
    def add_mem(self, state: np.ndarray, action: int, reward: float, features: tuple[bool]):
        self.memory.add(state, reward, action, features)
    
    def add_mem_bulk_unsafe(self, state: np.ndarray, action: np.ndarray[int], reward: np.ndarray[float], 
                            features: np.ndarray[bool]):
        self.memory.bulk_add_unsafe(state, reward, action, features)
    
    def add_mem_bulk(self, state: np.ndarray, action: np.ndarray[int], reward: np.ndarray[float], 
                            features: np.ndarray[bool]):
        self.memory.bulk_add(state, reward, action, features)
    
    def save_models(self, epoch: int):
        current = dict()
        current["memory"] = self.memory
        current["stat_dict_act"] = self.act_net.state_dict()
        current["stat_dict_nav"] = self.nav_net.state_dict()
        current["hyper_param"] = (self.action_num, self.memory.max_size, 
                                  self.discount, self.act_wd, self.nav_wd,
                                  self.eps, self.eps_decay, self.eps_min)
        torch.save(current, model_savepath %(epoch))
        del current
        
    def load_models(self, epoch: int, inference: bool=True):
        current = torch.load(model_savepath %(epoch))
        self.act_net.load_state_dict(current["stat_dict_act"])
        self.nav_net.load_state_dict(current["stat_dict_nav"])
        self.action_num, self.memory.max_size, self.discount, self.act_wd, \
            self.nav_wd,self.eps, self.eps_decay, self.eps_min = current["hyper_param"]
            
        if not inference:
            self.act_optimizer = self.optimizer(self.act_net.parameters(), 
                                                lr=self.lr, weight_decay=self.act_wd)
            self.nav_optimizer = self.optimizer(self.nav_net.parameters(), 
                                                lr=self.lr, weight_decay=self.nav_wd)
            self.memory = current["memory"]
            self.rng = default_rng()
        else:
            self.act_net.eval()
            self.nav_net.eval()
        
        del current
        torch.cuda.empty_cache()
    
    def train(self, batch_size: int=10, feature_loss_factor: float=100):
        indices = self.memory.replay_p(batch_size)
        for i in indices:
            start, end  = i-self.history_len, i+2
            frames      = self.memory.frames[start:end, :, :, :]
            rewards     = self.memory.rewards[start:end]
            actions     = self.memory.actions[start:end]
            is_combat   = self.memory.features[start:end, 0]
            is_right_dir= self.memory.features[start:end, 1]
            ongoing_mask= ~self.memory.features[start:end, 2]
            
            for s_start in range(0, self.padding_len):
                # We are updating the state (s_end-1) here
                # s_start to s_end-2 are the observation history
                # s_end-1 is the current state
                # s_end is the next state
                s_end = s_start + self.padding_len
                current = s_end-1
                
                # Distinguish between combat and navigation task
                if is_combat[current]:
                    model   = self.act_net
                    feature = is_combat.copy()
                    optimizer = self.act_optimizer
                    current_action = actions[current]
                else:
                    model   = self.nav_net
                    feature = is_right_dir.copy()
                    optimizer = self.nav_optimizer
                    current_action = actions[current] % 8
                
                # Predict game features and action values for the next state
                with torch.no_grad():
                    next_state = torch.from_numpy(frames[s_start+1:s_end+1, :, :, :]).float().cuda()
                    pred_feature = model.inf_feature(next_state)[:, 0]
                    pred_action = torch.max(model.inf_action()[-1]).item()
                
                # Calculate loss for game features
                true_feature = torch.from_numpy(feature[s_start+1:s_end+1]).float().cuda()
                game_feature_loss = self.criterion(true_feature, pred_feature) * feature_loss_factor
                
                # q-target = reward + discount * max_(a')(q(s',a'))
                q_target = torch.tensor(
                    rewards[current] + (self.discount * pred_action) if ongoing_mask[s_end-1] else rewards[current],
                    dtype=torch.float32, device=self.device)
                
                # Predict action values for the current state
                model.inf_feature(torch.from_numpy(frames[s_start:s_end, :, :, :]).float().cuda())
                pred_action = model.inf_action()[-1, current_action]
                
                # Calculate loss for action values
                action_value_loss = self.criterion(q_target, pred_action)
                optimizer.zero_grad()
                (action_value_loss + game_feature_loss).backward()
                optimizer.step()
        self.eps = self.eps * self.eps_decay if self.eps > self.eps_min else self.eps