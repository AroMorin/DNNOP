"""Environment class for detecting objects."""

from .environment import Environment
import torch

class Object_Detection(Environment):
    def __init__(self, env_params):
        super(Object_Detection, self).__init__(env_params)
        env_params = self.ingest_params_lvl1(env_params)
        self.train = env_params["train"]
        self.img_w = env_params["image width"]
        self.img_h = env_params["image height"]
        self.num_channels = env_params["number of channels"]
        self.path = env_params["path"]
        # List of images, each image is an NxM matrix
        self.images = []
        # List of sets, each set has shape (x, y)
        self.boxes = []
        self.validation_size = env_params["holdout size"]

    def ingest_params_lvl1(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "train": True,
                            "image width": 490,
                            "image height": 326,
                            "number of channels": 1,
                            "data path": "C:/Users/aaa2cn/Documents/phone_data/find_phone",
                            "holdout size": 5
                            }
        default_params.update(env_params)
        return default_params

    def load_images(self):
        
