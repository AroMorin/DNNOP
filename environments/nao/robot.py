"""NAO robot class"""

from ..environment import Environment
import naoqi
from naoqi import ALProxy

class Robot(Environment):
    def __init__(self, env_params):
        self.ingest_params(env_params)
        super().__init__(env_params["precision"])
        self.robot = True
        self.ip = env_params["ip"]
        self.port = env_params["port"]

    def ingest_params(self, env_params):
        assert type(env_params) is dict
        if "ip" not in env_params:
            env_params["ip"] = "localhost"
        if "port" not in env_params:
            env_params["port"] = 58463
        if "precision" not in env_params:
            env_params["precision"] = None

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

    def getJoints(self, joints=[]):
        if not hasattr(self, 'motion'):
            self.motion = ALProxy("ALMotion", self.ip, self.port)
        if len(joints)==0:
            joints = "Body"
        useSensors = False
        angles = self.motion.getAngles(joints, useSensors)
        return angles

    def setJoints(self, joints=[]):
        if not hasattr(self, 'motion'):
            self.motion = ALProxy("ALMotion", self.ip, self.port)
        if len(joints)==0:
            joints = "Body"
        useSensors = False
        angles = self.motion.getAngles(joints, useSensors)
        return angles
