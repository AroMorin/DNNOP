"""Environment class for detecting objects."""

from ..environment import Environment
import torch
import os
import csv
from PIL import Image

class Object_Detection(Environment):
    """Class contains methods relevant to the object detection envrionment."""
    def __init__(self, env_params):
        super(Object_Detection, self).__init__(env_params)
        env_params = self.ingest_params_lvl1(env_params)
        self.train = env_params["train"]
        self.img_w = env_params["image width"]
        self.img_h = env_params["image height"]
        self.channels = env_params["channels"]
        self.img_mode = env_params["image mode"]
        self.path = env_params["path"]
        self.nb_imgs = env_params["number of images"]
        # List of dictionaries, each entry has image name and data tensor
        self.data = []
        # List of images, as tensors
        self.images = []
        # Tensor of images (aggregation of self.images into a tensor)
        self.observation = None
        # List of sets, each set has shape (x, y)
        self.labels = []
        #self.scaled_centers = []
        self.validation_size = env_params["holdout size"]
        self.load_data()

    def ingest_params_lvl1(self, env_params):
        """Ingests parameters and provides defaults in case user did not provide
        any.
        """
        assert type(env_params) is dict
        default_params = {
                            "train": True,
                            "image width": 490,
                            "image height": 326,
                            "image mode": "L",  # L: Greyscale, RGB: Color
                            "channels": 1,
                            "data path": "C:/Users/aaa2cn/Documents/phone_data/find_phone/",
                            "holdout size": 5,
                            "number of images": 100,
                            "score type": "score"
                            }
        default_params.update(env_params)
        return default_params

    def step(self):
        """This is the step function. It loads an image for the pool to operate
        on.
        """
        pass

    def load_data(self):
        """Loads images from path folder and labels, and then links them
        in the images attribute.
        """
        self.load_images()
        self.load_labels()
        assert len(self.images) == len(self.labels)  # Sanity check
        for image in self.images:
            for label in self.labels:
                if image["name"] == label[0]:
                    image["center"] = [label[1], label[2]]  # Extend dict with label
                    self.labels.remove(label)
                    break  # Terminate inner loop
        assert len(self.labels) == 0  # Sanity check
        self.parse_data()

    def load_images(self):
        """Loads images, turns them to tensors, reshapes them and finally adds
        them to the images attribute.
        """
        my_files = os.listdir(self.path)
        for fn in my_files:
            if ".jpg" in fn:
                img_path = self.path+fn
                img = Image.open(img_path)
                img = img.convert(mode=self.img_mode)
                img_data = list(img.getdata())
                im_arr = torch.tensor(img_data, dtype=self.precision)
                im_arr = im_arr.reshape(shape=(1, self.channels, self.img_h, self.img_w))
                im_arr = im_arr.to(self.device)
                im_dict = {
                            "name":fn,
                            "data":im_arr
                            }
                self.images.append(im_dict)

    def load_labels(self):
        """Reads the labels file, parses contents and updates the labels
        attribute.
        """
        with open(self.path+"labels.txt") as f:
            for line in f:
                self.labels.append(line.split())

    def parse_data(self):
        """We will traverse the data attribute list and load the images and
        labels "in order", so as to ensure we can process by index.
        """
        self.data = self.images
        self.images = []
        self.labels = []
        self.names = []
        for item in self.data:
            self.images.append(item["data"])
            self.labels.append(item["center"])
            self.names.append(item["name"])
        self.observation = torch.cat(self.images)

    def evaluate(self, centers):
        total_err = 0
        correct = 0
        for i, center in enumerate(centers):
            total_err = total_err + self.compute_error(center, self.labels[i])
            correct += self.compute_acc(center, self.labels[i])
        acc = (correct/len(self.images))*100
        print ("Accuracy: %f" %acc)
        return total_err

    def compute_error(self, center, label):
        x = center[0].item()
        y = center[1].item()
        x_t = float(label[0])
        y_t = float(label[1])
        error = abs(x-x_t)+abs(y-y_t)
        return error

    def compute_acc(self, center, label):
        x = center[0].item()
        y = center[1].item()
        x_t = float(label[0])
        y_t = float(label[1])
        d_x = abs(x-x_t)
        d_y = abs(y-y_t)
        tolerance = 0.05
        if d_x<=tolerance and d_y<=tolerance:
            return 1
        else:
            return 0






















#
