from ..environment import Environment
import gym
import torch

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

    def ingest_params_lvl1(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "env name": "MsPacman-v0",
                            "scoring type": "score",
                            "populations": False  # Population-based optimization
                            }
        default_params.update(env_params)  # Update with user selections
        return default_params

    def init_env(self, name):
        self.env = gym.make(name)  # Location
        self.action_space = self.env.action_space
        self.obs_space = self.env.observation_space
        self.obs_high = self.obs_space.high
        self.obs_low = self.obs_space.low
        observation = self.env.reset()
        self.observation = torch.Tensor(observation).cuda()

    def step(self, action):
        """Instantiates the plotter class if a plot is requested by the user."""
        action = action.argmax().int()
        action = action.cpu().numpy()
        observation, reward, self.done, self.info = self.env.step(action)
        self.reward += reward
        self.observation = torch.Tensor(observation).cuda()
        self.iteration += 1

    def evaluate(self, inference):
        return torch.Tensor([self.reward])

    def reset_state(self):
        self.iteration = 0
        observation = self.env.reset()
        self.observation = torch.Tensor(observation).cuda()
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
