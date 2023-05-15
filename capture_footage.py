import vizdoom as vzd
import numpy as np
import os
import shutil

from tqdm import trange
from PIL import Image

from models import CombatAgent

# ============================== What is this ========================================
# Helper functions for capturing testing footage for RL agent into gif files
# ====================================================================================

def capture(agent: CombatAgent, game: vzd.vizdoom.DoomGame, repeat: int,
            action_space: list, nav_action_space: list, downsampler, 
            res: tuple[int, int]=(128, 72), episodes_to_capture: int=20, 
            cheat: bool=False):
    """Captures testing footage for RL agent and save into gif files.
    It would create a folder in root directory of doombot named after the agent.

    Args:
        agent (CombatAgent): the RL agent.
        game (vzd.vizdoom.DoomGame): vizdoom game instance.
        repeat (int): the number of frames to repeat actions for.
        action_space (list): action space for the combat model.
        nav_action_space (list): action space for the navigation model.
        downsampler (function): downsampling algorithm to use.
        res (tuple[int, int], optional): downsampling algorithm's target resolution. Defaults to (128, 72).
        episodes_to_capture (int, optional): number of episodes to capture. Defaults to 20.
        cheat (bool, optional): whether to inform agent of enemy presense, for internal usage, don't modify. Defaults to False.
    """
    
    if not cheat:
        agent.eval()
        
    name = agent.name
    repeat_iterator = [0]*repeat
    
    if cheat:
        name = name + "_cheat"
    
    if os.path.exists(name):
        shutil.rmtree(name)
    
    os.mkdir(name)
    
    for i in trange(episodes_to_capture):
        terminated = False
        game.new_episode()
        frames = []
        state, frame_lr = get_frame(game, downsampler, res, frames)
        while not terminated:
            if cheat:
                is_combat = check_for_enemies(state)
                action = agent.decide_move(frame_lr, is_combat)
            else:
                action, is_combat = agent.decide_move_blind(frame_lr)
            action = action_space[action] if is_combat else nav_action_space[action]
            for _ in repeat_iterator:
                game.make_action(action)
                if terminated := game.is_episode_finished():
                    reward = game.get_total_reward()
                    break
                state, frame_lr = get_frame(game, downsampler, res, frames)
        duration = 1000 / 35
        frames[0].save(f"{name}/{int(reward)}_{i:02d}.gif", save_all = True, 
                       append_images=frames[1:], optimize=True, duration=duration)
    
    # also save a cheat verion
    if not cheat:
        capture(agent, game, repeat, action_space, nav_action_space, downsampler, 
                res, episodes_to_capture, cheat=True)

def get_frame(game: vzd.vizdoom.DoomGame, downsampler, res: tuple, 
              frames: list) -> np.ndarray:
    """Fetch new frame for the current state, capture it, then returns downsampled frame

    Args:
        game (vzd.vizdoom.DoomGame): vizdoom game instance.
        downsampler (_type_): downsampling algorithm to use.
        res (tuple): downsampling algorithm's target resolution.
        frames (list): list for holding all captured frames in this episode

    Returns:
        np.ndarray: downsampled frame
    """    
    state = game.get_state()
    frame = state.screen_buffer
    frame_lr = downsampler(frame, res)
    frames.append(Image.fromarray(frame.transpose(1,2,0)))
    return (state, frame_lr)

def check_for_enemies(state) -> bool:
    """Check for the presence of enemies in current frame

    Args:
        state (state): current state

    Returns:
        bool: whether enemies are present
    """    
    enemies = {"Zombieman", "ShotgunGuy", "MarineChainsawVzd", "ChaingunGuy", "Demon", "HellKnight"}
    for label in state.labels:
        if label.object_name in enemies:
            return True
    return False
    