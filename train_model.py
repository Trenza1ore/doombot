# Imports
import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import vizdoom as vzd
import matplotlib.pyplot as plt
from tqdm import trange
from models import model_savepath, CombatAgent, DRQNv1
from discord_webhook import discord_bot
from vizdoom_utils import *

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

def train_agent(game: vzd.vizdoom.DoomGame, nav_game: vzd.vizdoom.DoomGame, 
                agent: CombatAgent, action_space: list, nav_action_space: list, 
                episode_to_watch: int, skip_training: bool=False, 
                plot: bool=False, discord: bool=False, epoch_num: int=3, 
                frame_repeat: int=4, epoch_step: int=1000, load_epoch: int=-1, 
                downsampler=resize_cv_linear, res=(108, 60)
                ) -> tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]:
    
    all_scores = [[], []]
    train_quartiles = [[], [], []]
    if (not skip_training):
        bot = discord_bot()
        start = time()
        if load_epoch < 0:
            epoch_start = 0
        else:
            epoch_start = load_epoch
            #agent.load_q_net(epoch_start, inference=False)
        
        # Messy but function calls can be expensive
        print("Initial filling of replay memory with random actions")
        for _ in trange(50):
            if game.is_episode_finished():
                game.new_episode()
            state = game.get_state()
            frame = downsampler(state.screen_buffer, res)
            action = agent.rng.integers(0, len(action_space))
            reward = game.make_action(action_space[action], frame_repeat)
            terminated = game.is_episode_finished()
            game_variables = state.game_variables
            reward = reward + 20 * game_variables[0] + 1000 * game_variables[1]
            is_combat = False
            is_right_direction = False
            for label in state.labels:
                if label.object_name in ("Zombieman", "ShotgunGuy", "ChaingunGuy"):
                    is_combat = True
                    break
            else:
                for label in state.labels:
                    if label.object_name == "GreenArmor":
                        is_right_direction = True
                        break
            agent.add_mem(frame, action, reward, (is_combat, is_right_direction, terminated))
        
        for epoch in range(epoch_start, epoch_num):
            game.new_episode()
            train_scores = []
            train_kills = []
            print(f"\n==========Epoch {epoch+1}==========")
            terminated = False
            
            for _ in trange(epoch_step):
                state = game.get_state()
                game_variables = state.game_variables
                
                # counter-intuitively, this is actually faster than creating a list of names
                is_combat = False
                is_right_direction = False
                for label in state.labels:
                    if label.object_name in ("Zombieman", "ShotgunGuy", "ChaingunGuy"):
                        is_combat = True
                        break
                else:
                    for label in state.labels:
                        if label.object_name == "GreenArmor":
                            is_right_direction = True
                            break
                
                # game variables: [health, kill count]
                action = agent.decide_move(frame, is_combat)
                reward = game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat)
                if is_combat:
                    reward = 30 * game_variables[0] + 1000 * game_variables[1]
                else:
                    reward += 30 * game_variables[0]
                agent.add_mem(frame, action, reward, (is_combat, is_right_direction, terminated))
                agent.train()
                
                terminated = game.is_episode_finished()
                
                if terminated:
                    train_scores.append(game.get_total_reward())
                    train_kills.append(game_variables[1])
                    game.new_episode()
            
            # Save statistics
            all_scores[0].extend(train_scores)
            all_scores[1].extend(train_kills)
            train_scores = np.array(train_scores)
            Q1, Q2, Q3 = np.percentile(train_scores, 25), np.median(train_scores), np.percentile(train_scores, 75)
            
            train_quartiles[0].append(Q1)
            train_quartiles[1].append(Q2)
            train_quartiles[2].append(Q3)
            
            if plot:
                plt.bar(np.arange(len(train_scores)), train_scores, 1)
                plt.title(f"Epoch {epoch+1}, eps = {agent.eps}")
                plt.xlim(0, len(train_scores)+1)
                plt.savefig(f"plots/{epoch}.png")
                plt.clf()
                plt.bar(np.arange(len(all_scores[0])), all_scores[0], 1)
                plt.xlim(0, len(all_scores[0])+1)
                plt.savefig(f"plots/{epoch}a.png")
                plt.clf()
                quartile_x = np.arange(len(train_quartiles[0])) + epoch_start + 1
                plt.plot(quartile_x, train_quartiles[0], 'r--', label="Q1")
                plt.plot(quartile_x, train_quartiles[1], 'k-', label="Q2")
                plt.plot(quartile_x, train_quartiles[2], 'b--', label="Q3")
                plt.title("Quartiles")
                plt.legend()
                plt.xlim(0, len(train_quartiles[0])+1)
                plt.savefig("plots/train_quartiles.png")
                plt.clf()
                plt.bar(np.arange(len(all_scores[1])), all_scores[1], 1)
                plt.title("Kill Count")
                plt.xlim(0, len(all_scores[1])+1)
                plt.ylim(-1, 7)
                plt.savefig("plots/train_kill_counts.png")
                plt.clf()
            
            
            stats = f"Result:\nmean: {np.mean(train_scores):.2f} +- {train_scores.std():.2f}\nQ1: {np.percentile(train_scores, 25)}\nQ2: {np.median(train_scores)}\nQ3: {np.percentile(train_scores, 75)}\nmin: {train_scores.min():.2f}\nmax: {train_scores.max():.2f}"
            print(stats)
            duration = int(time()-start);
            timer = f"{duration//60:d} min {duration%60:d} sec"
            print(timer)
            
            if discord:
                bot.send_stat(stats+'\n'+timer)
            
            if discord:
                bot.send_img(epoch)
            
            np.save("train_quartiles.npy", np.asfarray(train_quartiles))
            np.save("train_kill_counts.npy", np.asfarray(all_scores[1]))
            np.save(f"scores_{epoch}.npy", train_scores)
            np.save(f"scores_all_{epoch}.npy", np.asfarray(all_scores[0]))
            
            print("==========Nav Train==========")
            
            terminated = False
            nav_game.new_episode()
            for _ in trange(epoch_step//4):
                state = nav_game.get_state()
                frame = downsampler(state.screen_buffer, res)
                action = agent.decide_move_nav(frame)
                game_variables = state.game_variables
                
                # counter-intuitively, this is actually faster than creating a list of names
                is_combat = False
                is_right_direction = False
                for label in state.labels:
                    if label.object_name == "GreenArmor":
                        is_right_direction = True
                        break
                
                reward = nav_game.make_action(nav_action_space[action], frame_repeat)
                agent.add_mem(frame, action, reward, (is_combat, is_right_direction, terminated))
                agent.train()
                
                terminated = nav_game.is_episode_finished()
                
                if terminated:
                    nav_game.new_episode()
            
            # Save models after epoch
            agent.save_models(epoch)
            
    game.close()
    
    response = 'y' if skip_training else ''
    while response not in ['y', 'n']:
        response = input("Continue to watch? (y/n): ")
    if response == 'y':
        game = create_game(combat_config, color=True, label=True, res=(256, 144), visibility=True)
        agent.eps = agent.eps_min
        sleep(1.0)
        for _ in range(episode_to_watch):
            game.new_episode()
            while not game.is_episode_finished():
                state = game.get_state()
                frame = downsampler(state.screen_buffer, res)
                action = agent.decide_move_blind(frame)
                terminated = game.is_episode_finished()
                game_variables = state.game_variables

            # Sleep between episodes
            sleep(1.0)
            reward = game.get_total_reward()
            reward = reward + 20 * game_variables[0] + 1000 * game_variables[1]
            print(f"Reward: {reward}, Kills: {int(game_variables[1])}, Health: {int(game_variables[0])}")
    return (agent, game, all_scores)
            
def main():
    game = create_game(combat_config, color=True, label=True, res=(256, 144), visibility=True)
    nav_game = create_game(navigation_config, color=True, label=True, res=(256, 144), visibility=True)
    DEVICE = check_gpu()
    n = game.get_available_buttons_size()
    act_actions = [list(a) for a in it.product([False, True], repeat=n)]
    nav_actions = [[False]+list(a) for a in it.product([False, True], repeat=n-1)]
    agent = CombatAgent(
        device=DEVICE, mem_size=100_000, action_num=len(act_actions), nav_action_num=len(nav_actions), 
        discount=0.99, lr=0.01, loss=nn.MSELoss, act_wd=0.001, nav_wd=0.001, optimizer=optim.SGD, state_len=10,
        act_model=DRQNv1, nav_model=DRQNv1, eps=1., eps_decay=0.9995, eps_min=0.1, seed=21013377)
    print(agent.action_num, agent.nav_action_num)
    
    _, _, scores = train_agent(game=game, nav_game=nav_game, agent=agent, action_space=act_actions, 
                               nav_action_space=nav_actions, episode_to_watch=10, skip_training=False, 
                               plot=True, discord=True, epoch_num=100, frame_repeat=5, epoch_step=500, 
                               load_epoch=-1)
    plt.plot(scores)
    plt.show()

main()
