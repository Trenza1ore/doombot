# Imports
import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
import vizdoom as vzd
import matplotlib.pyplot as plt
from tqdm import trange
from models.legacy import DQNv2, Double_DQN_agent, DQN_agent, model_savepath
from vizdoom_utils import create_game
from discord_webhook import discord_bot

config_file_path = os.path.join(vzd.scenarios_path, "testing.cfg")
#config_file_path = os.path.join(vzd.scenarios_path, "not_so_deadly_corridor.cfg")

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

def train_agent(game: vzd.vizdoom.DoomGame, agent: DQN_agent, action_space: np.ndarray, episode_to_watch: int, 
                skip_training: bool=False, plot: bool=False, discord: bool=False, epoch_num: int=3, frame_repeat: int=4, epoch_step: int=1000, load_epoch: int=-1) -> tuple[DQN_agent, vzd.vizdoom.DoomGame, list[float]]:
    
    all_scores = []
    epoch_quartiles = [[], [], []]
    if (not skip_training):
        bot = discord_bot()
        start = time()
        if load_epoch < 0:
            epoch_start = 0
        else:
            epoch_start = load_epoch
            agent.load_q_net(epoch_start, inference=False)
        for epoch in range(epoch_start, epoch_num):
            game.new_episode()
            train_scores = []
            gstep = 0
            print(f"Epoch {epoch+1}")
            
            for _ in trange(epoch_step, leave=False):
                state = game.get_state()
                state = [state.screen_buffer, state.depth_buffer]
                action = agent.decide_move(state)
                reward = game.make_action(action_space[action], frame_repeat)
                # Retrieve the shaping reward
                shaping_reward = game.get_game_variable(vzd.GameVariable.USER1)
                #print(f"Reward: {reward}, Shaping: {shaping_reward}")
                terminated = game.is_episode_finished()
                
                if not terminated:
                    next_state = game.get_state()
                    next_state = [next_state.screen_buffer, next_state.depth_buffer]
                    #next_state = [next_state.screen_buffer[0,:,:], next_state.screen_buffer[1,:,:], next_state.screen_buffer[2,:,:], next_state.depth_buffer]
                    #depth_buffer = np.expand_dims(next_state.depth_buffer, axis=0)
                    #next_state = np.concatenate((next_state.screen_buffer, depth_buffer), axis=0)
                else:
                    next_state = [np.zeros((240, 320))]*2
                
                agent.add_mem(state, action, reward+shaping_reward, next_state, terminated)
                
                if gstep > agent.batch_size:
                    # if epoch == 0:
                    #     agent.first_epoch_train()
                    # else:
                    #     agent.train()
                    agent.train()
                
                if terminated:
                    train_scores.append(game.get_total_reward())
                    game.new_episode()
                
                gstep += 1
            
            #agent.update_double_net()
            agent.save_q_net(epoch)
            all_scores.extend(train_scores)
            train_scores = np.array(train_scores)
            
            Q1, Q2, Q3 = np.percentile(train_scores, 25), np.median(train_scores), np.percentile(train_scores, 75)
            
            epoch_quartiles[0].append(Q1)
            epoch_quartiles[1].append(Q2)
            epoch_quartiles[2].append(Q3)
            
            if plot:
                plt.bar(np.arange(len(train_scores)), train_scores, 1)
                plt.title(f"Epoch {epoch+1}, eps = {agent.eps}")
                plt.xlim(0, len(train_scores)+1)
                plt.ylim(-7000,7000)
                plt.savefig(f"plots/{epoch}.png")
                plt.clf()
                plt.bar(np.arange(len(all_scores)), all_scores, 1)
                plt.xlim(0, len(all_scores)+1)
                plt.ylim(-7000,7000)
                plt.savefig(f"plots/{epoch}a.png")
                plt.clf()
                quartile_x = np.arange(len(epoch_quartiles[0])) + epoch_start + 1
                plt.plot(quartile_x, epoch_quartiles[0], 'r--', label="Q1")
                plt.plot(quartile_x, epoch_quartiles[1], 'k-', label="Q2")
                plt.plot(quartile_x, epoch_quartiles[2], 'b--', label="Q3")
                plt.title("Epoch Medians")
                plt.legend()
                plt.xlim(0, len(epoch_quartiles[0])+1)
                plt.savefig(f"plots/epoch_quartiles.png")
                plt.clf()
            
            if discord:
                bot.send_img(epoch)
            
            np.save("epoch_quartiles.npy", np.asfarray(epoch_quartiles))
            print(f"Result:\nmean: {np.mean(train_scores):.2f} +- {train_scores.std():.2f}\nQ1: {np.percentile(train_scores, 25)}\nQ2: {np.median(train_scores)}\nQ3: {np.percentile(train_scores, 75)}\nmin: {train_scores.min():.2f}\nmax: {train_scores.max():.2f}")
            duration = int(time()-start);
            print(f"Took {duration//60:d} min {duration%60:d} sec\n\n")
            
    game.close()
    
    response = 'y' if skip_training else ''
    while response not in ['y', 'n']:
        response = input("Continue to watch? (y/n): ")
    if response == 'y':
        game = create_game(config_file_path, color=False, visibility=True)
        agent.eps = agent.eps_min
        sleep(1.0)
        for _ in range(episode_to_watch):
            game.new_episode()
            while not game.is_episode_finished():
                state = game.get_state()
                # state = [magnitude(Sobel(state.screen_buffer, CV_64F, dx=1, dy=0), Sobel(state.screen_buffer, CV_64F, dx=0, dy=1)), 
                #          state.depth_buffer]
                state = [state.screen_buffer, state.depth_buffer]
                #best_action_index = agent.decide_move(state)
                # print(best_action_index)
                # game.set_action(action_space[best_action_index])
                # for _ in range(frame_repeat):
                #     game.advance_action()
                action = agent.decide_move(state)
                print(action_space[action])
                game.make_action(action_space[action], frame_repeat)

            # Sleep between episodes
            sleep(1.0)
            score = game.get_total_reward()
            print("Total score: ", score)
    return (agent, game, all_scores)
            
            
            
            
def main():
    game = create_game(config_file_path, color=False, visibility=True)
    DEVICE = check_gpu()
    n = game.get_available_buttons_size()
    action_space = [list(a) for a in it.product([0, 1], repeat=n)]
    agent = DQN_agent(device=DEVICE, action_num=len(action_space), mem_size=40000, batch_size=50, discount=0.99, lr=0.01, wd=0.001, eps_decay=0.99998, eps_min=0.01)
    
    #agent.load_q_net(16, inference=True)
    print(agent.q_net)
    print(n, len(action_space))
    _, _, scores = train_agent(game=game, agent=agent, action_space=action_space, 
                               episode_to_watch=10, skip_training=False, plot=True, 
                               discord=False, epoch_num=40, frame_repeat=5, 
                               epoch_step=5000, load_epoch=8)
    #_, _, scores = train_agent(game=game, agent=agent, action_space=action_space, episode_to_watch=10, plot=True, discord=False, epoch_num=20, frame_repeat=5, epoch_step=5000)
    plt.plot(scores)
    plt.show()

main()
