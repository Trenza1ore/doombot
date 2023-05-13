import vizdoom as vzd
import numpy as np
from tqdm import trange
from models import CombatAgent
from PIL import Image
import os
import shutil

def capture(agent: CombatAgent, game: vzd.vizdoom.DoomGame, repeat: int,
            action_space: list, nav_action_space: list, downsampler, 
            res=(128, 72), episodes_to_capture: int=20):
    agent.eval()
    name = agent.name
    repeat_iterator = [0]*repeat
    
    if os.path.exists(name):
        shutil.rmtree(name)
    os.mkdir(name)
    
    for i in trange(episodes_to_capture):
        terminated = False
        game.new_episode()
        frames = []
        frames_low_res = []
        frame_lr = get_frame(game, downsampler, res, frames, frames_low_res)
        while not terminated:
            action, is_combat = agent.decide_move_blind(frame_lr)
            action = action_space[action] if is_combat else nav_action_space[action]
            for _ in repeat_iterator:
                game.make_action(action)
                if terminated := game.is_episode_finished():
                    reward = game.get_total_reward()
                    break
                frame_lr = get_frame(game, downsampler, res, frames, frames_low_res)
        duration = 1000 / 35
        frames[0].save(f"{name}/{int(reward)}_{i:02d}.gif", save_all = True, append_images=frames[1:], optimize=True, duration=duration)
        frames_low_res[0].save(f"{name}/{int(reward)}_{i:02d}_low_res.gif", save_all = True, append_images=frames_low_res[1:], optimize=True, duration=duration)

def get_frame(game: vzd.vizdoom.DoomGame, downsampler, res: tuple, 
              frames: list, frames_low_res: list) -> np.ndarray:
    state = game.get_state()
    frame = state.screen_buffer
    frame_lr = downsampler(frame, res)
    frames.append(Image.fromarray(frame.transpose(1,2,0)))
    frames_low_res.append(Image.fromarray(frame_lr.transpose(1,2,0)))
    return frame_lr