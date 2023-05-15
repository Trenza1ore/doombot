import vizdoom as vzd

from time import sleep, time
from tqdm import trange
from random import randrange
from capture_footage import capture

from models import *
from stats import *
from vizdoom_utils import *

# ============================== What is this ========================================
# Helper functions for running training sessions for an RL agent
# ====================================================================================

class inf_ep_max:
    """a class created to represent max value as Python 3's int has no such thing
    """
    def __eq__(self, __value: object) -> bool:
        return False
    def __gt__(self, __value: object) -> bool:
        return True
    def __ge__(self, __value: object) -> bool:
        return True
    def __lt__(self, __value: object) -> bool:
        return False
    def __le__(self, __value: object) -> bool:
        return False

def train_agent(game: vzd.vizdoom.DoomGame, 
                agent: CombatAgent, action_space: list, nav_action_space: list, 
                skip_training: bool=False, discord: bool=False, epoch_num: int=3, 
                frame_repeat: int=4, epoch_step: int=500, load_epoch: int=-1, 
                downsampler=resize_cv_linear, res: tuple[int, int]=(128, 72), 
                random_runs: bool=False, ep_max: int=0, save_interval: int=0
                ) -> tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]:
    """Runs a training session of an RL agent, with discord webhook support for sending statistics of training scores after each epoch of training.
    Written in an unstructured manner to have marginal (yet existing) performance gains. Deadly corridor is better trained with train_agent_corridor.
    
    Args:
        game (vzd.vizdoom.DoomGame): vizdoom game instance.
        agent (CombatAgent): the RL agent to train.
        action_space (list): action space for the combat model.
        nav_action_space (list): action space for the navigation model.
        skip_training (bool, optional): whether to skip training (for debug only). Defaults to False.
        discord (bool, optional): whether to have discord webhook send training stats and plots after every epoch. Defaults to False.
        epoch_num (int, optional): number of epoches to train (not necessarily reached if ep_max is set). Defaults to 3.
        frame_repeat (int, optional): the number of frames to repeat. Defaults to 4.
        epoch_step (int, optional): minimum number of steps in an epoch (the last episode in an epoch is always finished). Defaults to 500.
        load_epoch (int, optional): legacy option, use the load_models method of agent instance instead. Defaults to -1.
        downsampler (_type_, optional): downsampling algorithm to use, bilinear intepolation is the most balanced but nearest is slightly faster. Defaults to resize_cv_linear.
        res (tuple[int, int], optional): downsampling algorithm's target resolution. Defaults to (128, 72).
        random_runs (bool, optional): whether to include short, purely randomized episodes between epoches, minimum steps is set to 500. Defaults to False.
        ep_max (int, optional): maximum episodes for this training session, checked after every epoch and overwrites epoch_num, 0=disable. Defaults to 0.
        save_interval (int, optional): automatically save the models and replay memory every save_interval epoches, 0=disable. Defaults to 0.

    Returns:
        tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]: returns the agent, game instance and training scores
    """    
    
    if save_interval == 0:
        save_interval = epoch_num+1
    
    if ep_max <= 0:
        ep_max = inf_ep_max()
        
    all_scores = [[], []]
    train_quartiles = [[], [], [], []]
    if (not skip_training):
        bot = discord_bot(extra=f"deathmatch '{agent.name}' lr={agent.lr:.8f}")
        try:
            start = time()
            if load_epoch < 0:
                epoch_start = 0
            else:
                epoch_start = load_epoch
            
            action_space_size = len(action_space)
            nav_action_space_size = len(nav_action_space)
            enemies = {"Zombieman", "ShotgunGuy", "MarineChainsawVzd", "ChaingunGuy", "Demon", "HellKnight"}
            
            print("Initial filling of replay memory with random actions")
            terminated = False
            health = 100.
            game.new_episode()
            
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
                        
                        terminated = game.is_episode_finished()
                
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
                
                stats = plot_stat(train_scores, all_scores, train_quartiles, epoch, agent, bot, epoch_start)
                
                duration = int(time()-start);
                timer = f"{duration//60:d} min {duration%60:d} sec"
                print(timer)
                
                if discord:
                    bot.send_string(stats+'\n'+timer)
                    bot.send_img(epoch)
                
                # Save models after epoch
                agent.save_models(epoch) if not (epoch+1)%save_interval else None
                
                if len(all_scores[0]) > ep_max:
                    agent.save_models(epoch) if (epoch+1)%save_interval else None
                    bot.send_string("Training Complete!")
                    capture(agent, game, frame_repeat, action_space, nav_action_space, downsampler, res, 10)
                    game.close()
                    return
                
                sleep(60) # pause for a minute to recover from heat
        except Exception as e:
            bot.send_error(e)
            game.close()
            return
    
    bot.send_string("Training Complete!")        
    game.close()
    return (agent, game, all_scores)

def train_agent_corridor(game: vzd.vizdoom.DoomGame, nav_game: vzd.vizdoom.DoomGame,
                agent: CombatAgent, action_space: list, nav_action_space: list, 
                skip_training: bool=False, discord: bool=False, epoch_num: int=3, 
                frame_repeat: int=4, epoch_step: int=500, load_epoch: int=-1, 
                downsampler=resize_cv_linear, res: tuple[int, int]=(128, 72), 
                nav_runs: bool=False, ep_max: int=0, save_interval: int=0
                ) -> tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]:
    """Runs a training session of an RL agent, with discord webhook support for sending statistics of training scores after each epoch of training.
    Written in an unstructured manner to have marginal (yet existing) performance gains. Used specifically for training in deadly corridor.
    
    Args:
        game (vzd.vizdoom.DoomGame): vizdoom game instance.
        agent (CombatAgent): the RL agent to train.
        action_space (list): action space for the combat model.
        nav_action_space (list): action space for the navigation model.
        skip_training (bool, optional): whether to skip training (for debug only). Defaults to False.
        discord (bool, optional): whether to have discord webhook send training stats and plots after every epoch. Defaults to False.
        epoch_num (int, optional): number of epoches to train (not necessarily reached if ep_max is set). Defaults to 3.
        frame_repeat (int, optional): the number of frames to repeat. Defaults to 4.
        epoch_step (int, optional): minimum number of steps in an epoch (the last episode in an epoch is always finished). Defaults to 500.
        load_epoch (int, optional): legacy option, use the load_models method of agent instance instead. Defaults to -1.
        downsampler (_type_, optional): downsampling algorithm to use, bilinear intepolation is the most balanced but nearest is slightly faster. Defaults to resize_cv_linear.
        res (tuple[int, int], optional): downsampling algorithm's target resolution. Defaults to (128, 72).
        nav_runs (bool, optional): whether to include short, navigation-only training between epoches, minimum steps is set to 500. Defaults to False.
        ep_max (int, optional): maximum episodes for this training session, checked after every epoch and overwrites epoch_num, 0=disable. Defaults to 0.
        save_interval (int, optional): automatically save the models and replay memory every save_interval epoches, 0=disable. Defaults to 0.

    Returns:
        tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]: returns the agent, game instance and training scores
    """    
    
    if save_interval == 0:
        save_interval = epoch_num+1
        
    if ep_max <= 0:
        ep_max = inf_ep_max()
    
    all_scores = [[], []]
    train_quartiles = [[], [], [], []]
    if (not skip_training):
        bot = discord_bot(extra=f"deadly corridor '{agent.name}' lr={agent.lr:.8f}")
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
                
                print(f"\n==========Epoch {epoch+1:3d}==========")
                
                if nav_runs:
                    print("\nNav run")
                    terminated = False
                    nav_game.new_episode()
                    
                    for _ in trange(500):
                        state = nav_game.get_state()
                        frame = downsampler(state.screen_buffer, res)
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        reward = -5.

                        for label in state.labels:
                            if label.object_name == "GreenArmor":
                                reward = 10.
                                break
                        
                        # game variables: [health]
                        action = randrange(0, nav_action_space_size)
                        reward += nav_game.make_action(nav_action_space[action], frame_repeat)
                        agent.add_mem(frame, action, reward, (False, terminated))
                        agent.nav_train()
                        
                        if (terminated := nav_game.is_episode_finished()):
                            nav_game.new_episode()
                    
                    while not terminated:
                        state = nav_game.get_state()
                        frame = downsampler(state.screen_buffer, res)
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        reward = -5.
                        for label in state.labels:
                            if label.object_name == "GreenArmor":
                                reward = 10.
                                break
                        
                        # game variables: [health, kill count]
                        action = randrange(0, nav_action_space_size)
                        reward = nav_game.make_action(nav_action_space[action], frame_repeat)
                        
                        agent.add_mem(frame, action, reward, (False, terminated))
                        terminated = nav_game.is_episode_finished()
                
                train_scores = []
                train_kills = []
                
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
                    
                    if (terminated := game.is_episode_finished()):
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
                
                stats = plot_stat(train_scores, all_scores, train_quartiles, epoch, agent, bot, epoch_start)
                
                duration = int(time()-start);
                timer = f"{duration//60:d} min {duration%60:d} sec"
                print(timer)
                
                if discord:
                    bot.send_string(stats+'\n'+timer)
                    bot.send_img(epoch)
                
                # Save models after epoch
                agent.save_models(epoch) if not (epoch+1)%save_interval else None
                
                if len(all_scores[0]) > ep_max:
                    agent.save_models(epoch) if (epoch+1)%save_interval else None
                    bot.send_string("Training Complete!")
                    capture(agent, game, frame_repeat, action_space, nav_action_space, downsampler, res, 10)
                    nav_game.close()
                    game.close()
                    return
                
                sleep(45) # pause for 45 seconds to recover from heat
        except Exception as e:
            bot.send_error(e)
            nav_game.close()
            game.close()
            return
    nav_game.close()
    game.close()
    return (agent, game, all_scores)
