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
from training_procedure import *

navigation_config = os.path.join(vzd.scenarios_path, "empty_corridor.cfg")
combat_config4 = os.path.join(vzd.scenarios_path, "deadly_corridor_4.cfg")
combat_config = os.path.join(vzd.scenarios_path, "deadly_corridor_hugo.cfg")
combat_config2 = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
combat_config3 = os.path.join(vzd.scenarios_path, "deathmatch_hugo.cfg")

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
    for config in [combat_config, combat_config2, combat_config4, combat_config3]:
        lr_range = [0.000002, 0.00001, 0.00005] if "corridor" in config else [0.000025]
        for lr in lr_range:
            DEVICE = check_gpu()
            game = create_game(config, color=True, label=True, res=(256, 144), visibility=True)
            n = game.get_available_buttons_size()
            act_actions = [list(a) for a in it.product([False, True], repeat=n)]
            nav_actions = [[False]+list(a)+[False]*3 for a in it.product([False, True], repeat=3)]
            
            # Remove the idle action in both action space
            # Ensure that the agent always turn left or right in navigation mode
            act_actions = act_actions[1:]
            nav_actions = [action for action in nav_actions if action[2] or action[3]]
            
            print(f"Action space: {len(act_actions)} (combat), {len(nav_actions)} (nav)")
            
            # Not using PSO, too slow
            pso_config = {
                "inertial_weight"   : 0.5,
                "num_particles"     : 100,
                "max_param_value"   : 500,
                "min_param_value"   : -500
            }
            
            agent = CombatAgent(
                device=DEVICE, mem_size=250_000, action_num=len(act_actions), nav_action_num=len(nav_actions), 
                discount=0.99, lr=lr, dropout=0.5, loss=nn.HuberLoss, act_wd=0, nav_wd=0, optimizer=Adam, 
                state_len=10, act_model=DRQNv2, nav_model=DQNv1, eps=1., eps_decay=0.99995, eps_min=0.1)
            
            if "corridor" in config:
                nav_game = create_game(navigation_config, color=True, label=True, res=(256, 144), visibility=True)
                train_agent_corridor(game=game, agent=agent, nav_game=nav_game, action_space=act_actions,
                                nav_action_space=nav_actions, episode_to_watch=10, skip_training=False, 
                                plot=True, discord=True, epoch_num=20, frame_repeat=4, epoch_step=5000, 
                                load_epoch=-1, res=(128, 72), nav_runs=True)
            else:
                train_agent(game=game, agent=agent, action_space=act_actions, nav_action_space=nav_actions, 
                            episode_to_watch=10, skip_training=False, plot=True, discord=True, epoch_num=18, 
                            frame_repeat=4, epoch_step=5000, load_epoch=-1, res=(128, 72), random_runs=False)

            sleep(300) # pause for 5 minutes to recover from heat

main()
