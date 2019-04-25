"""Pacman environment"""
from .gym_base import Gym_base

class MsPacman(Gym_base):
    def __init__(self, env_params):
        super(MsPacman, self).__init__(env_params)
        env_params = self.ingest_params_lvl2(env_params)
        self.rendering = env_params['render']
        self.RAM = env_params['RAM']
        self.init_game()

    def ingest_params_lvl2(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "render": False,
                            "RAM": True
                            }
        default_params.update(env_params)  # Update with user selections
        return default_params

    def init_game(self):
        if self.RAM:
            self.init_env("MsPacman-ram-v0")
        else:
            self.init_env("MsPacman-v0")







        #