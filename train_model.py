# Imports
import itertools as it
import os
import numpy as np
import torch
import torch.nn as nn
import vizdoom as vzd

from time import sleep, time
from tqdm import trange
from random import randrange
from torch.optim import SGD, Adam, Adagrad
from torch_pso import ChaoticPSO, ParticleSwarmOptimizer as PSO

from models import *
from discord_webhook import discord_bot
from vizdoom_utils import *
from stat_plotter import plot_stat
from training_procedure import train_agent

combat_config = os.path.join(vzd.scenarios_path, "deadly_corridor_hugo.cfg")
navigation_config = os.path.join(vzd.scenarios_path, "empty_corridor.cfg")

def check_gpu() -> torch.device:
    # Uses GPU if available
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print("\nCUDA device detected.")
    else:
        DEVICE = torch.device("cpu")
        print("\nNo CUDA device detected.")
    return DEVICE
            
def main():
    game = create_game(combat_config, color=True, label=True, res=(256, 144), visibility=True)
    nav_game = create_game(navigation_config, color=True, label=True, res=(256, 144), visibility=True)
    DEVICE = check_gpu()
    n = game.get_available_buttons_size()
    act_actions = [list(a) for a in it.product([False, True], repeat=n)]
    nav_actions = [[False]+list(a) for a in it.product([False, True], repeat=n-1)]
    
    # Remove the idle action in both action space
    # Ensure that the agent always move forward in navigation mode
    act_actions = act_actions[1:]
    nav_actions = [action for action in nav_actions if action[1]]
    
    print(f"Action space: {len(act_actions)} (combat), {len(nav_actions)} (nav)")
    
    # Not using PSO, too slow
    pso_config = {
        "inertial_weight"   : 0.5,
        "num_particles"     : 100,
        "max_param_value"   : 500,
        "min_param_value"   : -500
    }
    
    agent = CombatAgent(
        device=DEVICE, mem_size=100_000, action_num=len(act_actions), nav_action_num=len(nav_actions), 
        discount=0.99, lr=0.1, loss=nn.HuberLoss, act_wd=0, nav_wd=0, optimizer=Adam, state_len=10,
        act_model=DRQNv2, nav_model=DQNv1, eps=1., eps_decay=0.99996, eps_min=0.1)
    
    _, _, scores = train_agent(game=game, nav_game=nav_game, agent=agent, action_space=act_actions, 
                               nav_action_space=nav_actions, episode_to_watch=10, skip_training=False, 
                               plot=True, discord=True, epoch_num=100, frame_repeat=4, epoch_step=5000, 
                               load_epoch=-1, res=(128, 72))

main()
