import vizdoom as vzd

def create_game(config_path: str, color: bool=False, depth: bool=True, label: bool=False, 
                automap: bool=False, res: tuple[int, int]=(256, 144), visibility: bool=False, *args, **kwargs) -> vzd.vizdoom.DoomGame:
    
    # create an instance of doom game
    print("Initializing Doom... ", end='')
    game = vzd.DoomGame()
    
    # remove named arguments set to boolean False or anything representing that
    for key, value in kwargs.items():
        if not value:
            del kwargs[key]
    
    # load scenario's default configuration
    game.load_config(config_path)
    
    # set game to sync/async player mode or async spectator mode
    game.set_mode(vzd.Mode.PLAYER) if "async" not in kwargs else \
        (game.set_mode(vzd.Mode.ASYNC_PLAYER) if "spec" not in kwargs else \
            game.set_mode(vzd.Mode.ASYNC_SPECTATOR))
    
    # set rendering options
    exec(f"game.set_screen_resolution(vzd.ScreenResolution.RES_{res[0]:d}X{res[1]:d})")
    exec(f"game.set_screen_format(vzd.ScreenFormat.{'CRCGCB' if color else 'GRAY8'})")
    game.set_window_visible(visibility)
    
    # no need to set buffer access if human is playing
    if "spec" not in kwargs:
        # set "cheating" buffer access
        game.set_depth_buffer_enabled(depth)
        game.set_labels_buffer_enabled(label)
        game.set_automap_buffer_enabled(automap)
        if "automap_config" in kwargs:
            config = kwargs["automap_config"]
            game.set_automap_mode(config["mode"])
            game.set_automap_rotate(config["rotate"])
            game.set_automap_render_textures(config["texture"])
    
    # finished initializing the game
    game.init()
    print("done")
    return game