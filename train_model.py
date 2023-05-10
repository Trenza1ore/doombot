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
                frame_repeat: int=4, epoch_step: int=500, load_epoch: int=-1, 
                downsampler=resize_cv_linear, res=(108, 60)
                ) -> tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]:
    all_scores = [[], []]
    train_quartiles = [[], [], [], []]
    if (not skip_training):
        bot = discord_bot()
        try:
            start = time()
            if load_epoch < 0:
                epoch_start = 0
            else:
                epoch_start = load_epoch
            
            action_space_size = len(action_space)
            
            for epoch in range(epoch_start, epoch_num):
                
                # Messy but function calls can be expensive
                print("Filling of replay memory with random actions")
                terminated = False
                kills = 0.
                health = 100.
                for _ in trange(500):
                    state = game.get_state()
                    game_variables = state.game_variables
                    frame = downsampler(state.screen_buffer, res)
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    is_right_direction = False
                    for label in state.labels:
                        if label.object_name in ("Zombieman", "ShotgunGuy", "ChaingunGuy"):
                            reward = 0.
                            is_combat = True
                            break
                    else:
                        reward = -500.
                        for label in state.labels:
                            if label.object_name == "GreenArmor":
                                reward = 500.
                                is_right_direction = True
                                break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    enemy_killed = game_variables[1] - kills
                    health = game_variables[0]
                    kills = game_variables[1]
                    
                    # game variables: [health, kill count]
                    action = randrange(0, action_space_size)
                    reward = game.make_action(action_space[action], frame_repeat)
                    reward -= 5 * health_lost      # penalty for losing health
                    reward += 100 * enemy_killed   # reward for killing enemy
                    
                    agent.add_mem(frame, action, reward, (is_combat, is_right_direction, terminated))
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        kills = 0.
                        health = 100.
                        game.new_episode()
                
                for _ in trange(500):
                    state = game.get_state()
                    game_variables = state.game_variables
                    frame = downsampler(state.screen_buffer, res)
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    is_right_direction = False
                    for label in state.labels:
                        if label.object_name in ("Zombieman", "ShotgunGuy", "ChaingunGuy"):
                            reward = 0.
                            is_combat = True
                            break
                    else:
                        reward = -500.
                        for label in state.labels:
                            if label.object_name == "GreenArmor":
                                reward = 500.
                                is_right_direction = True
                                break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    enemy_killed = game_variables[1] - kills
                    health = game_variables[0]
                    kills = game_variables[1]
                    
                    # game variables: [health, kill count]
                    action = 15
                    reward = game.make_action(action_space[action], frame_repeat)
                    reward -= 5 * health_lost      # penalty for losing health
                    reward += 100 * enemy_killed   # reward for killing enemy
                    
                    agent.add_mem(frame, action, reward, (is_combat, is_right_direction, terminated))
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        kills = 0.
                        health = 100.
                        game.new_episode()
                
                game.new_episode()
                train_scores = []
                train_kills = []
                print(f"\n==========Epoch {epoch+1}==========")
                terminated = False
                
                kills = 0.
                health = 100.
                
                for _ in trange(epoch_step):
                    state = game.get_state()
                    frame = downsampler(state.screen_buffer, res)
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    is_right_direction = False
                    for label in state.labels:
                        if label.object_name in ("Zombieman", "ShotgunGuy", "ChaingunGuy"):
                            reward = 0.
                            is_combat = True
                            break
                    else:
                        reward = -500.
                        for label in state.labels:
                            if label.object_name == "GreenArmor":
                                reward = 500.
                                is_right_direction = True
                                break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    enemy_killed = game_variables[1] - kills
                    health = game_variables[0]
                    kills = game_variables[1]
                    
                    # game variables: [health, kill count]
                    action = agent.decide_move(frame, is_combat)
                    reward = game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat)
                    reward -= 5 * health_lost      # penalty for losing health
                    reward += 100 * enemy_killed   # reward for killing enemy
                    
                    agent.add_mem(frame, action, reward, (is_combat, is_right_direction, terminated))
                    agent.train()
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        kills = 0.
                        health = 100.
                        train_scores.append(game.get_total_reward())
                        train_kills.append(game_variables[1])
                        game.new_episode()
                
                # Save statistics
                all_scores[0].extend(train_scores)
                all_scores[1].extend(train_kills)
                
                stats = plot_stat(train_scores, all_scores, train_quartiles, epoch, agent, bot, epoch_start, plot)
                
                duration = int(time()-start);
                timer = f"{duration//60:d} min {duration%60:d} sec"
                print(timer)
                
                if discord:
                    bot.send_string(stats+'\n'+timer)
                    bot.send_img(epoch)
                
                np.save("scores/train_quartiles.npy", np.asfarray(train_quartiles))
                np.save("scores/train_kill_counts.npy", np.asfarray(all_scores[1]))
                np.save(f"scores/scores_{epoch}.npy", train_scores)
                np.save(f"scores/scores_all_{epoch}.npy", np.asfarray(all_scores[0]))
                
                # print("==========Nav Train==========")
                
                # terminated = False
                # nav_game.new_episode()
                # for _ in trange(epoch_step//4):
                #     state = nav_game.get_state()
                #     frame = downsampler(state.screen_buffer, res)
                #     action = agent.decide_move_nav(frame)
                #     game_variables = state.game_variables
                    
                #     # counter-intuitively, this is actually faster than creating a list of names
                #     is_combat = False
                #     is_right_direction = False
                #     for label in state.labels:
                #         if label.object_name == "GreenArmor":
                #             is_right_direction = True
                #             break
                    
                #     reward = nav_game.make_action(nav_action_space[action], frame_repeat)
                #     agent.add_mem(frame, action, reward, (is_combat, is_right_direction, terminated))
                #     agent.train()
                    
                #     terminated = nav_game.is_episode_finished()
                    
                #     if terminated:
                #         nav_game.new_episode()
                
                # Save models after epoch
                agent.save_models(epoch)
        except Exception as e:
            bot.send_error(e)
            game.close()
            return
            
    game.close()
    
    # response = 'y' if skip_training else ''
    # while response not in ['y', 'n']:
    #     response = input("Continue to watch? (y/n): ")
    if True:
        game = create_game(combat_config, color=True, label=True, res=(256, 144), visibility=True)
        agent.eps = agent.eps_min
        sleep(1.0)
        for i in range(episode_to_watch):
            game.new_episode()
            while not game.is_episode_finished():
                state = game.get_state()
                frame = downsampler(state.screen_buffer, res)
                action = agent.decide_move_no_nav(frame)
                terminated = game.is_episode_finished()
                np.save(f"ep{i:03d}-{i:03d}.npy", frame)

            reward = game.get_total_reward()
            print(f"Reward: {reward}, Kills: {int(state.game_variables[1])}, Health: {int(state.game_variables[0])}")
    return (agent, game, all_scores)

def train_agent_no_nav(game: vzd.vizdoom.DoomGame, nav_game: vzd.vizdoom.DoomGame, 
                agent: CombatAgent, action_space: list, nav_action_space: list, 
                episode_to_watch: int, skip_training: bool=False, 
                plot: bool=False, discord: bool=False, epoch_num: int=3, 
                frame_repeat: int=4, epoch_step: int=500, load_epoch: int=-1, 
                downsampler=resize_cv_linear, res=(108, 60)
                ) -> tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]:
    
    all_scores = [[], []]
    train_quartiles = [[], [], [], []]
    if (not skip_training):
        bot = discord_bot()
        try:
            start = time()
            if load_epoch < 0:
                epoch_start = 0
            else:
                epoch_start = load_epoch
            
            action_space_size = len(action_space)
            
            for epoch in range(epoch_start, epoch_num):
                
                print("Filling of replay memory with random actions")
                terminated = False
                kills = 0.
                health = 100.
                for _ in trange(500):
                    state = game.get_state()
                    game_variables = state.game_variables
                    frame = downsampler(state.screen_buffer, res)
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    for label in state.labels:
                        if label.object_name in ("Zombieman", "ShotgunGuy", "ChaingunGuy"):
                            reward = 0.
                            is_combat = True
                            break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    enemy_killed = game_variables[1] - kills
                    health = game_variables[0]
                    kills = game_variables[1]
                    
                    # game variables: [health, kill count]
                    action = randrange(0, action_space_size)
                    reward = game.make_action(action_space[action], frame_repeat)
                    reward += 100 * enemy_killed   # reward for killing enemy
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated))
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        kills = 0.
                        health = 100.
                        game.new_episode()
                        
                for _ in trange(500):
                    state = game.get_state()
                    game_variables = state.game_variables
                    frame = downsampler(state.screen_buffer, res)
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    for label in state.labels:
                        if label.object_name in ("Zombieman", "ShotgunGuy", "ChaingunGuy"):
                            reward = 0.
                            is_combat = True
                            break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    enemy_killed = game_variables[1] - kills
                    health = game_variables[0]
                    kills = game_variables[1]
                    
                    # game variables: [health, kill count]
                    action = 15
                    reward = game.make_action(action_space[action], frame_repeat)
                    reward += 100 * enemy_killed   # reward for killing enemy
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated))
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        kills = 0.
                        health = 100.
                        game.new_episode()
                
                game.new_episode()
                train_scores = []
                train_kills = []
                print(f"\n==========Epoch {epoch+1}==========")
                terminated = False
                
                kills = 0.
                health = 100.
                
                for _ in trange(epoch_step):
                    state = game.get_state()
                    frame = downsampler(state.screen_buffer, res)
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    for label in state.labels:
                        if label.object_name in ("Zombieman", "ShotgunGuy", "ChaingunGuy"):
                            reward = 0.
                            is_combat = True
                            break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    enemy_killed = game_variables[1] - kills
                    health = game_variables[0]
                    kills = game_variables[1]
                    
                    # game variables: [health, kill count]
                    action = agent.decide_move_no_nav(frame)
                    reward = game.make_action(action_space[action], frame_repeat)
                    reward += 100 * enemy_killed   # reward for killing enemy
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated))
                    agent.train()
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        kills = 0.
                        health = 100.
                        train_scores.append(game.get_total_reward())
                        train_kills.append(game_variables[1])
                        game.new_episode()
                
                # Save statistics
                all_scores[0].extend(train_scores)
                all_scores[1].extend(train_kills)
                
                stats = plot_stat(train_scores, all_scores, train_quartiles, epoch, agent, bot, epoch_start, plot)
                
                duration = int(time()-start);
                timer = f"{duration//60:d} min {duration%60:d} sec"
                print(timer)
                
                if discord:
                    bot.send_string(stats+'\n'+timer)
                    bot.send_img(epoch)
                
                # Save models after epoch
                agent.save_models(epoch)
        except Exception as e:
            bot.send_error(e)
            game.close()
            return
            
    game.close()
    
    # response = 'y' if skip_training else ''
    # while response not in ['y', 'n']:
    #     response = input("Continue to watch? (y/n): ")
    if True:
        game = create_game(combat_config, color=True, label=True, res=(256, 144), visibility=True)
        agent.eps = agent.eps_min
        sleep(1.0)
        for i in range(episode_to_watch):
            game.new_episode()
            while not game.is_episode_finished():
                state = game.get_state()
                frame = downsampler(state.screen_buffer, res)
                action = agent.decide_move_no_nav(frame)
                terminated = game.is_episode_finished()
                np.save(f"ep{i:03d}-{i:03d}.npy", frame)

            reward = game.get_total_reward()
            print(f"Reward: {reward}, Kills: {int(state.game_variables[1])}, Health: {int(state.game_variables[0])}")
    return (agent, game, all_scores)
            
def main():
    game = create_game(combat_config, color=True, label=True, res=(256, 144), visibility=True)
    nav_game = None#create_game(navigation_config, color=True, label=True, res=(256, 144), visibility=True)
    DEVICE = check_gpu()
    n = game.get_available_buttons_size()
    act_actions = [list(a) for a in it.product([False, True], repeat=n)]
    nav_actions = [[False]+list(a) for a in it.product([False, True], repeat=n-1)]
    
    # Remove the idle action in both action space
    # Ensure that the agent always move forward in navigation mode
    act_actions = act_actions[1:]
    nav_actions = [action for action in nav_actions if action[1]]
    
    print(len(act_actions), len(nav_actions))
    
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
    print(agent.action_num, agent.nav_action_num)
    
    _, _, scores = train_agent_no_nav(game=game, nav_game=nav_game, agent=agent, action_space=act_actions, 
                               nav_action_space=nav_actions, episode_to_watch=10, skip_training=False, 
                               plot=True, discord=True, epoch_num=100, frame_repeat=4, epoch_step=5000, 
                               load_epoch=-1, res=(128, 72))

main()
