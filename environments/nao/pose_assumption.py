"""NAO robot class"""

from .robot import Robot
import naoqi
from naoqi import ALProxy

class Pose_Assumption(Robot):
    def __init__(self, env_params):
        super(Pose_Assumption, self).__init__(env_params)
        env_params = self.ingest_params2(env_params)
        self.target = env_params["target error"]
        self.joints = env_params["joints to move"]
        self.target_angles = env_params["target anges"]
        self.default_pose = "LyingBack"
        self.penalty = 0  # State
        self.error = float('inf')  # State
        self.assume_pose(self.default_pose)
        self.set_stiffness()

    def ingest_params2(self, env_params):
        if "target error" not in env_params:
            env_params["target error"] = 0
        if "joints to move" not in env_params:
            env_params["joints to move"] = ["HeadYaw"]
        if "target angles" not in env_params:
            env_params["target angles"] = [0.0]
        return env_params

    def set_stiffness(self):
        time = 1.0  # Seconds
        value = 0.9  # Stiffness (max 1/min 0, higher is looser)
        self.motion.setStiffnesses(self.joints, value, time)

    def step(self):
        """In this function the robot will return to default pose, to
        be ready for the new command.
        """

    def reset_state(self):
        self.penalty = 0
        self.error = 0

    def evaluate(self, inference):

        angles = self.get_joints()
        self.calc_error(angles)

    def apply(self, inference):
        for idx, value in enumerate(inference):
            name = self.joints[idx]
            limits = self.motion.getLimits(name)
            min_angle = limits[0]
            max_angle = limits[1]
            max_vel = limits[2]  # Unenforced
            max_tor = limits[3]  # Unenforced
            value = self.cap_angle(value, min_angle, max_angle)
            inference[idx] = value
        self.set_joints(inference)

    def calc_error(self, angles):
        self.error = sum(list(map(add, angles.values(), self.targets)))
        print("Error: %f" %self.error)

    def cap_angle(self, x, a, b):
        penalty = 10  # Safety penalty
        if x<a:
            self.penalty += penalty
            x = a
        elif x>b:
            self.penalty += penalty
            x = b
        return x
















#
