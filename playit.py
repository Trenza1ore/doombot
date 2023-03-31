import vizdoom as vzd
import os

config_file_path = os.path.join(vzd.scenarios_path, "testing.cfg")

def create_game(config_path: str, color: bool = False, res: int = 240, visibility: bool = False) -> vzd.vizdoom.DoomGame:
    print("Initializing Doom... ", end='')
    game = vzd.DoomGame()
    game.load_config(config_path)
    game.set_window_visible(visibility)
    game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
    game.set_depth_buffer_enabled(True)
    game.set_render_hud(True)
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
        s = game.get_state()
        game.advance_action()