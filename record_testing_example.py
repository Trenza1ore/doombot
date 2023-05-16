import os
import vizdoom as vzd
import itertools as it
import torch.nn as nn

from torch.optim import Adam

from run import check_gpu
from models import CombatAgent, DRQNv2, DQNv1
from vizdoom_utils import capture, resize_cv_linear, create_game

# ============================== What is this ========================================
# An example script for loading a pre-trained RL agent and recording testing episodes
# ====================================================================================

config = os.path.join(vzd.scenarios_path, "deathmatch_hugo.cfg")
is_corridor = False
mem_size = 1
name = "dm5_1"

DEVICE = check_gpu()
game = create_game(config, color=True, label=True, res=(256, 144), visibility=False)
n = game.get_available_buttons_size()
act_actions = [list(a) for a in it.product([False, True], repeat=n)]
nav_actions = [[False]+list(a)+[False]*3 for a in it.product([False, True], repeat=3)]
act_actions = act_actions[1:]
nav_actions = [action for action in nav_actions if action[2] or action[3]]

agent = CombatAgent(
    device=DEVICE, mem_size=mem_size, action_num=len(act_actions), nav_action_num=len(nav_actions), 
    discount=0.99, lr=0, dropout=0.5, loss=nn.HuberLoss, act_wd=0, nav_wd=0, optimizer=Adam, 
    state_len=10, act_model=DRQNv2, nav_model=DQNv1, eps=1., eps_decay=0.99995, name=name)
print("Loading data from saved pth file... ", end='')

agent.load_models(19, name, True)
print("done")

capture(agent, game, 4, act_actions, nav_actions, resize_cv_linear, episodes_to_capture=30)