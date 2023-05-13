import vizdoom as vzd
import os
from matplotlib.image import imsave
from vizdoom_utils import resize_cv_linear, create_game

is_recording = True
res=(256, 144)
#res=(1920, 1080)
#config_file_path = os.path.join(vzd.scenarios_path, "deathmatch.cfg")
config_file_path = os.path.join(vzd.scenarios_path, "deathmatch_hugo.cfg")
#config_file_path = os.path.join(vzd.scenarios_path, "deadly_corridor_hugo.cfg")


if is_recording:
    if not os.path.isdir("screenshots"):
        os.mkdir("screenshots")
game = create_game(config_file_path, color=True, depth=False, res=res, visibility=True, asyn=True, spec=True)

while True:
    game.new_episode()
    i = 0
    while not game.is_episode_finished():
        i += 1
        state = game.get_state()
        if is_recording:
            frame = state.screen_buffer
            frame_lr = resize_cv_linear(frame, (128, 72))
            imsave(f"screenshots/{i}.png", frame.transpose(1,2,0))
            imsave(f"screenshots/low_res_{i}.png", frame_lr.transpose(1,2,0))
        game.advance_action()