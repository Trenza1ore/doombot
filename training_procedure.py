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

def train_agent(game: vzd.vizdoom.DoomGame, 
                agent: CombatAgent, action_space: list, nav_action_space: list, 
                episode_to_watch: int, skip_training: bool=False, 
                plot: bool=False, discord: bool=False, epoch_num: int=3, 
                frame_repeat: int=4, epoch_step: int=500, load_epoch: int=-1, 
                downsampler=resize_cv_linear, res=(108, 60), random_runs=False
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
            nav_action_space_size = len(nav_action_space)
            enemies = {"Zombieman","ShotgunGuy","MarineChainsawVzd","ChaingunGuy", "Demon", "HellKnight"}
            
            print("Initial filling of replay memory with random actions")
            terminated = False
            health = 100.
            
            for _ in trange(500):
                state = game.get_state()
                game_variables = state.game_variables
                frame = downsampler(state.screen_buffer, res)
                
                # counter-intuitively, this is actually faster than creating a list of names
                reward = -5.
                is_combat = False
                for label in state.labels:
                    if label.object_name in enemies:
                        reward = 5.
                        is_combat = True
                        break
                
                # calculate change in health and kill count
                health_lost = health - game_variables[0]
                health = game_variables[0]
                
                # game variables: [health]
                if is_combat:
                    action = randrange(0, action_space_size)
                    reward += game.make_action(action_space[action], frame_repeat)
                else: 
                    action = randrange(0, nav_action_space_size)
                    reward += game.make_action(nav_action_space[action], frame_repeat)
                reward -= health_lost # negative reward for losing health
                
                agent.add_mem(frame, action, reward, (is_combat, terminated))
                
                terminated = game.is_episode_finished()
                
                if terminated:
                    health = 100.
                    game.new_episode()
            
            for epoch in range(epoch_start, epoch_num):
                
                if random_runs:
                    print("Filling of replay memory with random actions")
                    terminated = False
                    health = 100.
                    
                    for _ in trange(500):
                        state = game.get_state()
                        game_variables = state.game_variables
                        frame = downsampler(state.screen_buffer, res)
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        reward = -5.
                        is_combat = False
                        for label in state.labels:
                            if label.object_name in enemies:
                                reward = 5.
                                is_combat = True
                                break
                        
                        # calculate change in health and kill count
                        health_lost = health - game_variables[0]
                        health = game_variables[0]
                        
                        # game variables: [health]
                        if is_combat:
                            action = randrange(0, action_space_size)
                            reward += game.make_action(action_space[action], frame_repeat)
                        else: 
                            action = randrange(0, nav_action_space_size)
                            reward += game.make_action(nav_action_space[action], frame_repeat)
                        reward -= health_lost # negative reward for losing health
                        
                        agent.add_mem(frame, action, reward, (is_combat, terminated))
                        
                        terminated = game.is_episode_finished()
                        
                        if terminated:
                            health = 100.
                            game.new_episode()
                    
                    while not terminated:
                        state = game.get_state()
                        frame = downsampler(state.screen_buffer, res)
                        game_variables = state.game_variables
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        is_combat = False
                        for label in state.labels:
                            if label.object_name in enemies:
                                reward = 0.
                                is_combat = True
                                break
                        
                        # calculate change in health and kill count
                        health_lost = health - game_variables[0]
                        health = game_variables[0]
                        
                        # game variables: [health, kill count]
                        if is_combat:
                            action = randrange(0, action_space_size)
                            reward += game.make_action(action_space[action], frame_repeat)
                        else: 
                            action = randrange(0, nav_action_space_size)
                            reward += game.make_action(nav_action_space[action], frame_repeat)
                        reward -= health_lost # negative reward for losing health
                        
                        agent.add_mem(frame, action, reward, (is_combat, terminated))
                
                game.new_episode()
                train_scores = []
                train_kills = []
                print(f"\n==========Epoch {epoch+1:3d}==========")
                
                terminated = False
                health = 100.
                
                for _ in trange(epoch_step):
                    state = game.get_state()
                    frame = downsampler(state.screen_buffer, res)
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    reward = -10.
                    is_combat = False
                    for label in state.labels:
                        if label.object_name in enemies:
                            reward = 10.
                            is_combat = True
                            break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    health = game_variables[0]
                    
                    # game variables: [health, kill count]
                    action = agent.decide_move(frame, is_combat)
                    reward += game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat)
                    reward -= health_lost # negative reward for losing health
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated))
                    agent.train()
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        health = 100.
                        train_scores.append(game.get_total_reward())
                        train_kills.append(game_variables[1])
                        game.new_episode()
                
                while not terminated:
                    state = game.get_state()
                    frame = downsampler(state.screen_buffer, res)
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    for label in state.labels:
                        if label.object_name in enemies:
                            reward = 0.
                            is_combat = True
                            break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    health = game_variables[0]
                    
                    # game variables: [health, kill count]
                    action = agent.decide_move(frame, is_combat)
                    reward = game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat)
                    reward -= health_lost # negative reward for losing health
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated))
                    agent.train()
                    
                    if (terminated := game.is_episode_finished()):
                        train_scores.append(game.get_total_reward())
                        train_kills.append(game_variables[1])
                
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
                
                # Save models after epoch
                #(agent.save_models(epoch) if not (epoch+1)%5 else None) if epoch_num > 30 else agent.save_models(epoch)
                
                sleep(60) # pause for a minute to recover from heat
        except Exception as e:
            bot.send_error(e)
            game.close()
            return
            
    game.close()
    return (agent, game, all_scores)

def train_agent_corridor(game: vzd.vizdoom.DoomGame, nav_game: vzd.vizdoom.DoomGame,
                agent: CombatAgent, action_space: list, nav_action_space: list, 
                episode_to_watch: int, skip_training: bool=False, 
                plot: bool=False, discord: bool=False, epoch_num: int=3, 
                frame_repeat: int=4, epoch_step: int=500, load_epoch: int=-1, 
                downsampler=resize_cv_linear, res=(108, 60), nav_runs=False
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
            nav_action_space_size = len(nav_action_space)
            enemies = {"Zombieman", "ShotgunGuy", "ChaingunGuy"}
            
            print("Initial filling of replay memory with random actions")
            game.new_episode()
            terminated = False
            health = 100.
            kills = 0.
            
            for _ in trange(500):
                state = game.get_state()
                game_variables = state.game_variables
                frame = downsampler(state.screen_buffer, res)
                
                # counter-intuitively, this is actually faster than creating a list of names
                reward = -5.
                is_combat = False
                for label in state.labels:
                    if label.object_name in enemies:
                        reward = 5.
                        is_combat = True
                        break
                else:
                    for label in state.labels:
                        if label.object_name == "GreenArmor":
                            reward = 10.
                            break
                
                # calculate change in health and kill count
                health_lost = health - game_variables[0]
                new_kill = game_variables[1] - kills
                health = game_variables[0]
                kills = game_variables[1]
                
                # game variables: [health]
                if is_combat:
                    action = randrange(0, action_space_size)
                    reward += game.make_action(action_space[action], frame_repeat)
                else: 
                    action = randrange(0, nav_action_space_size)
                    reward += game.make_action(nav_action_space[action], frame_repeat)
                reward -= health_lost # negative reward for losing health
                reward += 50 * new_kill
                agent.add_mem(frame, action, reward, (is_combat, terminated))
                
                terminated = game.is_episode_finished()
                
                if terminated:
                    health = 100.
                    kills = 0
                    game.new_episode()
            
            for epoch in range(epoch_start, epoch_num):
                
                if nav_runs:
                    print("Nav run")
                    terminated = False
                    nav_game.new_episode()
                    
                    for _ in trange(500):
                        state = nav_game.get_state()
                        frame = downsampler(state.screen_buffer, res)
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        reward = -5.
                        is_combat = False

                        for label in state.labels:
                            if label.object_name == "GreenArmor":
                                reward = 10.
                                break
                        
                        # game variables: [health]
                        action = randrange(0, nav_action_space_size)
                        reward += game.make_action(nav_action_space[action], frame_repeat)
                        agent.add_mem(frame, action, reward, (is_combat, terminated))
                        agent.nav_train()
                        terminated = nav_game.is_episode_finished()
                        
                        if terminated:
                            nav_game.new_episode()
                    
                    while not terminated:
                        state = nav_game.get_state()
                        frame = downsampler(state.screen_buffer, res)
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        reward = -5.
                        is_combat = False
                        for label in state.labels:
                            if label.object_name == "GreenArmor":
                                reward = 10.
                                break
                        
                        # game variables: [health, kill count]
                        action = randrange(0, nav_action_space_size)
                        reward = nav_game.make_action(nav_action_space[action], frame_repeat)
                        
                        agent.add_mem(frame, action, reward, (is_combat, terminated))
                
                train_scores = []
                train_kills = []
                print(f"\n==========Epoch {epoch+1:3d}==========")
                
                game.new_episode()
                terminated = False
                health = 100.
                kills = 0.
                
                for _ in trange(epoch_step):
                    state = game.get_state()
                    frame = downsampler(state.screen_buffer, res)
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    reward = -10.
                    is_combat = False
                    for label in state.labels:
                        if label.object_name in enemies:
                            reward = 5.
                            is_combat = True
                            break
                    else:
                        for label in state.labels:
                            if label.object_name == "GreenArmor":
                                reward = 10.
                                break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    new_kill = game_variables[1] - kills
                    health = game_variables[0]
                    kills = game_variables[1]
                    
                    # game variables: [health, kill count]
                    action = agent.decide_move(frame, is_combat)
                    reward += game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat)
                    reward -= health_lost # negative reward for losing health
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated))
                    agent.train()
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        health = 100.
                        kills = 0.
                        train_scores.append(game.get_total_reward())
                        train_kills.append(game_variables[1])
                        game.new_episode()
                
                while not terminated:
                    state = game.get_state()
                    frame = downsampler(state.screen_buffer, res)
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    for label in state.labels:
                        if label.object_name in enemies:
                            reward = 5.
                            is_combat = True
                            break
                    else:
                        for label in state.labels:
                            if label.object_name == "GreenArmor":
                                reward = 10.
                                break
                    
                    # calculate change in health and kill count
                    health_lost = health - game_variables[0]
                    new_kill = game_variables[1] - kills
                    health = game_variables[0]
                    kills = game_variables[1]
                    
                    # game variables: [health, kill count]
                    action = agent.decide_move(frame, is_combat)
                    reward = game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat)
                    reward -= health_lost # negative reward for losing health
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated))
                    agent.train()
                    
                    if (terminated := game.is_episode_finished()):
                        train_scores.append(game.get_total_reward())
                        train_kills.append(game_variables[1])
                
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
                
                # Save models after epoch
                #(agent.save_models(epoch) if not (epoch+1)%5 else None) if epoch_num > 30 else agent.save_models(epoch)
                
                sleep(60) # pause for a minute to recover from heat
        except Exception as e:
            bot.send_error(e)
            nav_game.close()
            game.close()
            return
    nav_game.close()
    game.close()
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
            enemies = {"Zombieman","ShotgunGuy","MarineChainsawVzd","ChaingunGuy", "Demon", "HellKnight"}
            
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
                        if label.object_name in enemies:
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
                        if label.object_name in enemies:
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
                        if label.object_name in enemies:
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
                #agent.save_models(epoch)
        except Exception as e:
            bot.send_error(e)
            game.close()
            return
            
    game.close()
    return (agent, game, all_scores)