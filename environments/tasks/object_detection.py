"""Environment class for detecting objects."""

from ..environment import Environment
import torch
import os
import csv
from PIL import Image
import time


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
        self.nb_imgs = 0
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
        if not env_params["inference"]:
            self.load_data()
        self.diff = 0
        self.correct=0
        self.total_err = 0
        self.tot_correct = 0
        self.tolerance = torch.tensor(0.05, dtype=self.precision, device=self.device)


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
                            "score type": "score",
                            "inference": False
                            }
        default_params.update(env_params)
        return default_params

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
            label = [float(item["center"][0]), float(item["center"][1])]
            self.labels.append(label)
            self.names.append(item["name"])
        self.observation = torch.cat(self.images)
        self.labels = torch.tensor(self.labels, dtype=self.precision).cuda()
        self.nb_imgs = len(self.images)

    def evaluate(self, centers):
        """ Implemented in CUDA to improve execution speed
        """
        labels = self.labels.expand_as(centers)
        _ =  centers.sub_(labels).abs_()
        x = _.sum(dim=2)
        y = x.sum(dim=1)
        return y
        #comps = torch.le(diff, self.tolerance)
        #truths = comps[:,0] & comps[:,1]
        #correct = truths.sum()
        #acc = torch.mul(torch.div(correct, self.nb_imgs), 100)
        #return err

    def compute_error(self, center, label):
        x = center[0]
        y = center[1]
        x_t = label[0]
        y_t = label[1]
        error = torch.abs(torch.sub(x,x_t))+torch.abs(torch.sub(y,y_t))
        return error

    def compute_acc(self, center, label):
        x = round(center[0].item(), 4)
        y = round(center[1].item(), 4)
        x_t = float(label[0])
        y_t = float(label[1])
        d_x = abs(x-x_t)
        d_y = abs(y-y_t)
        tolerance = 0.05
        if d_x<=tolerance and d_y<=tolerance:
            return 1
        else:
            return 0

    def get_image(self, path):
        img = Image.open(path)
        img = img.convert(mode=self.img_mode)
        img_data = list(img.getdata())
        im_arr = torch.tensor(img_data, dtype=self.precision)
        im_arr = im_arr.reshape(shape=(1, self.channels, self.img_h, self.img_w))
        im_arr = im_arr.to(self.device)
        return im_arr




















#
