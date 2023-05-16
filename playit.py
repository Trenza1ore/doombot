import vizdoom as vzd
import os
from matplotlib.image import imsave
from vizdoom_utils import resize_cv_linear, create_game

# ============================== What is this ========================================
# A small program that allows human players to play (and record) a scenario themselves
# ====================================================================================

# config
is_recording = False
#res = (256, 144)
res = (1920, 1080)
resize_res = (128, 72)

config_file_path = os.path.join(vzd.scenarios_path, "deathmatch_hugo.cfg")
#config_file_path = os.path.join(vzd.scenarios_path, "hanger_test.cfg")
#config_file_path = os.path.join(vzd.scenarios_path, "deadly_corridor_hugo.cfg")

# create folder for holding screenshots
if is_recording:
    if not os.path.isdir("screenshots"):
        os.mkdir("screenshots")
game = create_game(config_file_path, color=True, depth=False, res=res, visibility=True, asyn=True, spec=True)

# play the game as a human (while taking screenshots)
while True:
    game.new_episode()
    i = 0
    while not game.is_episode_finished():
        i += 1
        state = game.get_state()
        if is_recording:
            frame = state.screen_buffer
            frame_lr = resize_cv_linear(frame, )
            imsave(f"screenshots/{i}.png", frame.transpose(1,2,0))
            imsave(f"screenshots/low_res_{i}.png", frame_lr.transpose(1,2,0))
        game.advance_action()
    print(f"You scored: {game.get_total_reward():.2f}")