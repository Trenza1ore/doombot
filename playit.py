import vizdoom as vzd
import os

#config_file_path = os.path.join(vzd.scenarios_path, "defend_the_line.cfg")
#config_file_path = os.path.join(vzd.scenarios_path, "empty_corridor.cfg")
config_file_path = os.path.join(vzd.scenarios_path, "deadly_corridor_hugo.cfg")

def create_game(config_path: str, color: bool = False, res: int = 240, visibility: bool = False) -> vzd.vizdoom.DoomGame:
    print("Initializing Doom... ", end='')
    game = vzd.DoomGame()
    game.load_config(config_path)
    game.set_window_visible(visibility)
    game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_render_hud(True)
    game.set_doom_skill(1)
    if not color:
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
    match res:
        case 480:
            game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        case 1080:
            game.set_screen_resolution(vzd.ScreenResolution.RES_1920X1080)
        case _:
            pass # defaults to 320x240
    game.init()
    print("done")
    return game

game = create_game(config_file_path, res=480, visibility=True)
while True:
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        for label in state.labels:
            if label.object_name in ("Zombieman", "ShotgunGuy", "ChaingunGuy"):
                print('enemy')
        game.advance_action()