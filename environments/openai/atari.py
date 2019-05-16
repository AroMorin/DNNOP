"""Pacman environment"""
from .gym_base import Gym_base

class Atari(Gym_base):
    def __init__(self, env_params):
        super(Atari, self).__init__(env_params)
        env_params = self.ingest_params_lvl2(env_params)
        self.rendering = env_params['render']
        self.RAM = env_params['RAM']
        self.init_game(env_params["game name"])

    def ingest_params_lvl2(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "game name": "Breakout-ram-v0",
                            "render": False,
                            "RAM": True
                            }
        default_params.update(env_params)  # Update with user selections
        if "ram" in default_params["game name"]:
            assert default_params["RAM"] == True
        return default_params

    def init_game(self, game):
        self.init_env(game)





        #
