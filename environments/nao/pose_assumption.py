"""NAO robot class"""

from .robot import Robot
import naoqi
from naoqi import ALProxy

class Pose_Assumption(Robot):
    def __init__(self, env_params):
        super(Pose_Assumption, self).__init__(env_params)
        env_params = self.ingest_params2(env_params)
        self.target = env_params["target pose error"]
        self.default_pose = "LyingBack"

    def ingest_params2(self, env_params):
        if "target pose error" not in env_params:
            env_params["target pose error"] = 0
        return env_params

    def step(self):
        """In this function the robot will return to default pose, to
        be ready for the new command.
        """
        self.goToPosture(self.default_pose)
