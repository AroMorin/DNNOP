"""NAO robot class"""

from ..environment import Environment
import naoqi
from naoqi import ALProxy

class Robot(Environment):
    def __init__(self, precision, vars):
        super().__init__(precision)
        self.robot = True
        self.ip = "localhost"
        self.port = 58463
        self.set_vars(vars)

    def set_vars(self, vars):
        assert type(vars) is dict
        if "ip" in vars:
            self.ip = vars["ip"]
        if "port" in vars:
            self.port = vars["port"]

    def say(self, message):
        if not hasattr(self, 'tts'):
            self.tts = ALProxy("ALTextToSpeech", self.ip, self.port)
        tts.say(message)

    def assume_pose(self, pose):
        if not hasattr(self, 'motion'):
            self.motion = ALProxy("ALMotion", self.ip, self.port)
        if not hasattr(self, 'robot_posture'):
            self.robot_posture = ALProxy("ALRobotPosture", self.ip, self.port)
        self.motion.wakeUp()
        self.robot_posture.goToPosture(pose, 0.5)

    def rest(self):
        if not hasattr(self, 'motion'):
            self.motion = ALProxy("ALMotion", self.ip, self.port)
        self.motion.rest()
