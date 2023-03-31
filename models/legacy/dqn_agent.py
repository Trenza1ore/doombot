import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import sample, randrange
from collections import deque
from numpy.random import default_rng, randint
from numpy import expand_dims, ndarray, stack, arange, argmax, array, repeat, tile
from models.legacy.network import DQNv2, model_savepath

class DQN_agent:
    def __init__(self, device: torch.device, action_num: int, mem_size: int, 
                 batch_size: int, discount: float, lr: float, wd: float,
                 replay_mem_runs: int=10, model=DQNv2,
                 eps: float=1, eps_decay: float=0.99, eps_min: float=0.1):
        
        # store model hyper parameters
        self.device = device
        self.action_num = action_num
        self.batch_size = batch_size
        self.discount = discount
        self.lr = lr
        self.wd = wd
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.replay_mem_runs = replay_mem_runs
        self.rand_num = batch_size // replay_mem_runs
        self.mem_size = mem_size
        
        # initialize new model(s)
        self.q_net = model(action_num, batch_size, device).to(device)
        
        # initialize loss measurer, optimiser and replay memory queue
        self.criterion = nn.MSELoss()
        self.optimiser = optim.Adam(self.q_net.parameters(), lr=lr, weight_decay=wd)
        self.memory = deque(maxlen=mem_size)
        
        self.rng = default_rng()
        self.batch_row_idx = arange(batch_size)
    
    def decide_move(self, state: list[ndarray, ndarray]) -> torch.tensor:
        if self.rng.uniform() < self.eps:
            return self.rng.integers(0, self.action_num)
        else:
            state = torch.from_numpy(
                expand_dims(state, axis=0)
                ).float().to(self.device)
            return torch.argmax(self.q_net.inference(state)).item()
    
    def add_mem(self, state: list[ndarray, ndarray], action: int, reward: float, 
                next_state: list[ndarray, ndarray], terminated: bool):
        self.memory.append((state, action, reward, next_state, terminated))
    
    def save_q_net(self, epoch: int):
        current = dict()
        current["stat_dict"] = self.q_net.state_dict()
        current["hyper_param"] = (self.action_num, self.mem_size, self.batch_size, 
                                  self.discount, self.lr, self.wd, self.replay_mem_runs, 
                                  self.eps, self.eps_decay, self.eps_min)
        torch.save(current, model_savepath %(epoch))
        del current
    
    def load_q_net(self, epoch: int, inference: bool=True):
        current = torch.load(model_savepath %(epoch))
        self.q_net.load_state_dict(current["stat_dict"])
        self.action_num, self.mem_size, self.batch_size, self.discount, self.lr, self.wd, \
            self.replay_mem_runs, self.eps, self.eps_decay, self.eps_min = current["hyper_param"]
            
        if not inference:
            self.rand_num = self.batch_size // self.replay_mem_runs
            self.criterion = nn.MSELoss()
            self.optimiser = optim.Adam(self.q_net.parameters(), lr=self.lr, weight_decay=self.wd)
            self.memory = deque(maxlen=self.mem_size)
            self.rng = default_rng()
            self.batch_row_idx = arange(self.batch_size)
        else:
            self.q_net.eval()
        
        
        del current
        torch.cuda.empty_cache()
    
    def train(self):
        #batch = sample_mem(self.memory, self.replay_mem_runs, self.rand_num)
        batch = sample(self.memory, self.batch_size)
        batch = array(batch, dtype=object)
        
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
            next_state_values = self.q_net(next_states).cpu().data.numpy()[idx][on_going]
            
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
    
def sample_mem(memory, run_len, run_num):
    mem_max = len(memory)
    element_selected = randint(0, mem_max, run_num)
    element_selected.sort()
    element_selected = repeat(element_selected, run_len) + tile(arange(run_len), run_num)
    element_selected[element_selected >= mem_max] = mem_max - 1
    return [memory[i] for i in element_selected]