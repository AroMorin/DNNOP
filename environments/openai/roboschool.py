"""Pacman environment"""
from .gym_base import Gym_base

class Roboschool(Gym_base):
    def __init__(self, env_params):
        super(Roboschool, self).__init__(env_params)
        env_params = self.ingest_params_lvl2(env_params)
        self.rendering = env_params['render']
        self.discrete = env_params['Discrete']
        self.RAM = False
        self.IMG = False
        self.init_module(env_params["module name"])

    def ingest_params_lvl2(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "module name": "RoboschoolAnt-v1",
                            "render": False,
                            "RAM": False,
                            "IMG": False,
                            "Discrete": False
                            }
        default_params.update(env_params)  # Update with user selections
        return default_params

    def init_module(self, game):
        self.init_env(game)





        #
