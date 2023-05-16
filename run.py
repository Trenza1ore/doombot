import itertools as it
import os
import torch
import torch.nn as nn
import vizdoom as vzd

from time import sleep
from torch.optim import SGD, Adam, Adagrad

from models import *
from vizdoom_utils import *
from models.training_procedure import *

# ============================== What is this ========================================
# The main program to run training sessions
# ====================================================================================

# Deadly Corridor Scenarios
empty_corridor = os.path.join(vzd.scenarios_path, "empty_corridor.cfg")         # no enemy, starts with pistol
empty_corridor_mod = os.path.join(vzd.scenarios_path, "empty_corridor_hugo.cfg")# no enemy, starts with shotgun
corridor_og_4 = os.path.join(vzd.scenarios_path, "deadly_corridor_4.cfg")       # difficulty 4, starts with pistol
corridor_mod_5 = os.path.join(vzd.scenarios_path, "deadly_corridor_hugo.cfg")   # difficulty 5, starts with pistol
corridor_og_5 = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")         # difficulty 5, starts with shotgun

# Deathmatch Scenarios
deathmatch_mod_5 = os.path.join(vzd.scenarios_path, "deathmatch_hugo.cfg")      # my version
deathmatch_og_5 = os.path.join(vzd.scenarios_path, "deathmatch_og_texture.cfg") # with original texture

# Tips for config options
# save_validation should be switch to True at least in the first run to check the size of each save
# format for a task: 
# (scenario_config, learning rate, target number for episodes to train, save interval, unique id)

# config
save_validation = False 
tasks = [
    #(corridor_og_4, 0.000002, 1000, 0, "dc4_0"),
    #(corridor_og_4, 0.00001, 1000, 0, "dc4_1"),
    #(deathmatch_og_5, 0.00001, 100, 0, "dm5_og_1"),
    #(deathmatch_og_5, 0.000002, 100, 0, "dm5_og_2"),
    #(corridor_og_5, 0.00001, 1000, 0, "dc5_1"),
    #(corridor_mod_5, 0.000002, 2000, 10, "dcm_0_0"),
    #(corridor_mod_5, 0.000002, 2000, 10, "dcm_0_1"),
    (deathmatch_mod_5, 0.00001, 1000, 10, "dm5_1"),
    (deathmatch_mod_5, 0.000002, 1000, 10, "dm5_0")
]

def check_gpu() -> torch.device:
    """Checks the system to see if CUDA devices are available.
    Warning:
    cudnn.benchmark hurts performance if convolutional layers receives varying input shape

    Returns:
        torch.device: device for running PyTorch with
    """    
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
    global save_validation, tasks
    
    # ep_max (maximum training episode) would overwrite the epoch_num (number of epochs) setting
    for config, lr, ep_max, save_interval, name in tasks:
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
            empty_config = empty_corridor_mod if "hugo" in config else empty_corridor
            nav_game = create_game(empty_config, color=True, label=True, res=(256, 144), visibility=True)
        
        if save_validation:
            agent.name = 'test'
            file_name = agent.save_models(-1)
            file_size = os.path.getsize(file_name)
            os.remove(file_name)
            file_size /= (1024*1024*1024)
            agent.name = name
            save_validation = False
            print(f"Save validation passed, each save occupies {file_size:.2f} GiB of storage space")
        
        print(f"Memory capacity: {mem_size} | Action space: {len(act_actions)} (combat), {len(nav_actions)} (nav)")
        
        if is_corridor:
            train_agent_corridor(game=game, agent=agent, nav_game=nav_game, action_space=act_actions,
                            nav_action_space=nav_actions, skip_training=False, discord=True, 
                            epoch_num=1000, frame_repeat=4, epoch_step=5000, load_epoch=-1, res=(128, 72), 
                            nav_runs=True, ep_max=ep_max, save_interval=save_interval)
        else:
            train_agent(game=game, agent=agent, action_space=act_actions, nav_action_space=nav_actions, 
                        skip_training=False, discord=True, epoch_num=1000, frame_repeat=4, 
                        epoch_step=5000, load_epoch=-1, res=(128, 72), random_runs=False, ep_max=ep_max, 
                        save_interval=save_interval)

        sleep(300) # pause for 5 minutes to recover from heat

if __name__ == "__main__":
    main()
