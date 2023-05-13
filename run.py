# Imports
import itertools as it
import os
import torch
import torch.nn as nn
import vizdoom as vzd

from time import sleep
from torch.optim import SGD, Adam, Adagrad

from models import *
from vizdoom_utils import *
from training_procedure import *

empty_corridor = os.path.join(vzd.scenarios_path, "empty_corridor.cfg")
corridor_og_4 = os.path.join(vzd.scenarios_path, "deadly_corridor_4.cfg")
corridor_mod_5 = os.path.join(vzd.scenarios_path, "deadly_corridor_hugo.cfg")
corridor_og_5 = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
deathmatch_mod_5 = os.path.join(vzd.scenarios_path, "deathmatch_hugo.cfg")

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
    
    save_validation = False
    
    # ep_max (maximum training episode) would overwrite the epoch_num (number of epochs) setting
    
    for config, lr, ep_max, save_interval, name in [
        (corridor_og_4, 0.000002, 1000, 5, "dc4_0"),
        (corridor_og_5, 0.000002, 1000, 5, "dc5_0"),
        (corridor_og_4, 0.00001, 1000, 5, "dc4_1"),
        (corridor_mod_5, 0.000002, 10000, 15, "dcm_0"),
        (deathmatch_mod_5, 0.000002, 1000, 15, "dm5_0"),
        ]:
        
        DEVICE = check_gpu()
        game = create_game(config, color=True, label=True, res=(256, 144), visibility=True)
        n = game.get_available_buttons_size()
        act_actions = [list(a) for a in it.product([False, True], repeat=n)]
        nav_actions = [[False]+list(a)+[False]*3 for a in it.product([False, True], repeat=3)]
        
        # Remove the idle action in both action space
        # Ensure that the agent always turn left or right in navigation mode
        act_actions = act_actions[1:]
        nav_actions = [action for action in nav_actions if action[2] or action[3]]
        
        is_corridor = "corridor" in config
        mem_size = 100_000 if ((is_corridor and ep_max <= 1000) or (not is_corridor and ep_max <= 100)) else 300_000
        
        agent = CombatAgent(
            device=DEVICE, mem_size=mem_size, action_num=len(act_actions), nav_action_num=len(nav_actions), 
            discount=0.99, lr=lr, dropout=0.5, loss=nn.HuberLoss, act_wd=0, nav_wd=0, optimizer=Adam, 
            state_len=10, act_model=DRQNv2, nav_model=DQNv1, eps=1., eps_decay=0.99995, name=name)
        
        if is_corridor:
            nav_game = create_game(empty_corridor, color=True, label=True, res=(256, 144), visibility=True)
        
        if not save_validation:
            agent.name = 'test'
            file_name = agent.save_models(-1)
            file_size = os.path.getsize(file_name)
            os.remove(file_name)
            file_size /= (1024*1024*1024)
            agent.name = name
            save_validation = True
            print(f"Save validation passed, each save occupies {file_size:.2f} GiB of storage space")
        
        print(f"Memory capacity: {mem_size} | Action space: {len(act_actions)} (combat), {len(nav_actions)} (nav)")
        
        if is_corridor:
            train_agent_corridor(game=game, agent=agent, nav_game=nav_game, action_space=act_actions,
                            nav_action_space=nav_actions, skip_training=False, plot=True, discord=True, 
                            epoch_num=1000, frame_repeat=4, epoch_step=5000, load_epoch=-1, res=(128, 72), 
                            nav_runs=True, ep_max=ep_max, save_interval=save_interval)
        else:
            train_agent(game=game, agent=agent, action_space=act_actions, nav_action_space=nav_actions, 
                        skip_training=False, plot=True, discord=True, epoch_num=1000, frame_repeat=4, 
                        epoch_step=5000, load_epoch=-1, res=(128, 72), random_runs=False, ep_max=ep_max, 
                        save_interval=save_interval)

        sleep(300) # pause for 5 minutes to recover from heat

main()
