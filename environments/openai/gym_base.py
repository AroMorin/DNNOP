from ..environment import Environment
import gym
import roboschool
import torch
import numpy as np

class Gym_base(Environment):
    def __init__(self, env_params):
        super(Gym_base, self).__init__(env_params)
        env_params = self.ingest_params_lvl1(env_params)
        self.env = None
        self.action_space = None
        self.obs_space = None
        self.obs_high = None
        self.obs_low = None
        self.observation = None
        self.reward = 0.  # Matrix of function values
        self.done = False
        self.info = {}
        self.iteration = 0  # State
        self.minimize = False
        self.target = 999999   # Infinity, max score
        self.RAM = False
        self.IMG = False
        self.discrete = True

    def ingest_params_lvl1(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "env name": "MsPacman-v0",
                            "scoring type": "score",
                            "populations": False,  # Population-based optimization
                            "RAM": False,
                            "IMG": False,
                            "Discrete": True
                            }
        default_params.update(env_params)  # Update with user selections
        return default_params

    def init_env(self, name):
        self.env = gym.make(name)  # Location
        #self.env._max_episode_steps = 50000
        self.action_space = self.env.action_space
        self.obs_space = self.env.observation_space
        self.obs_high = self.obs_space.high
        self.obs_low = self.obs_space.low
        observation = self.env.reset()
        self.set_obs(observation)

    def set_obs_(self, x):
        if self.RAM:
            self.observation = torch.Tensor(x).cuda()
        else:
            x = np.moveaxis(x, -1, 0)
            x = torch.Tensor(x).cuda()
            self.observation = x.unsqueeze(0)

    def set_obs(self, x):
        self.observation = x

    def step(self, action):
        """Instantiates the plotter class if a plot is requested by the user."""
        #if self.discrete:
        #    action = action.argmax().int()
        #action = action.cpu().detach().numpy()
        if len(action.shape) == (0):
            action = np.expand_dims(action, 0)
        #action = self.env.action_space.sample()
        observation, reward, self.done, self.info = self.env.step(action)
        self.reward += reward
        self.set_obs(observation)
        self.iteration += 1

    def evaluate(self, _):
        return self.reward

    def reset_state(self):
        self.iteration = 0
        observation = self.env.reset()
        self.set_obs(observation)
        self.reward = 0.  # Matrix of function values
        self.done = False
        self.info = {}

    def get_random_action(self):
        action = self.action_space.sample()
        return action

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
