"""NAO robot class"""

from .robot import Robot
import naoqi
import torch

class Pose_Assumption(Robot):
    def __init__(self, env_params):
        super(Pose_Assumption, self).__init__(env_params)
        env_params = self.ingest_params2(env_params)
        self.target = env_params["target error"]
        self.joints = env_params["joints to move"]
        self.target_angles = env_params["target angles"]
        self.default_pose = "LyingBack"
        self.penalty = 0  # State
        self.error = float('inf')  # State
        self.assume_pose(self.default_pose)
        self.set_stiffness()

    def ingest_params2(self, env_params):
        if "target error" not in env_params:
            env_params["target error"] = 0
        if "joints to move" not in env_params:
            env_params["joints to move"] = ["HeadYaw", "HeadPitch",
                                            "RShoulderPitch","RShoulderRoll",
                                            "RElbowYaw", "RElbowRoll",
                                            "RWristYaw",
                                            "RHipYawPitch",
                                            "RHipRoll", "RHipPitch", "RKneePitch",
                                            "RAnklePitch", "RAnkleRoll",
                                            "LShoulderPitch","LShoulderRoll",
                                            "LElbowYaw", "LElbowRoll",
                                            "LWristYaw",
                                            "LHipYawPitch",
                                            "LHipRoll", "LHipPitch", "LKneePitch",
                                            "LAnklePitch", "LAnkleRoll"
                                            ]
            # NOTE: joints must be named individually
        if "target angles" not in env_params:
            env_params["target angles"] = [0.0, 0.153,
                                            0.66, 0.914,
                                            0.994, 0.721,
                                            0.08432,
                                            -0.512, -0.04,
                                            -0.8299, 0.317,
                                            0.288, -0.268, 0.99, 0.175, -1.234,
                                            -0.819, -1.286, -0.58287, 0.118,
                                            0.2899, -0.09, 0.6, -0.046
                                            ]
        return env_params

    def set_stiffness(self):
        time = 1.0  # Seconds
        value = 0.9  # Stiffness (max 1/min 0, higher is looser)
        self.motion.stiffnessInterpolation(self.joints, value, time)

    def step(self):
        """In this function the robot will return to default pose, to
        be ready for the new command.
        """
        origin = [0.4]  # Arbitrary input
        self.observation = torch.tensor(origin,
                                        dtype=self.precision,
                                        device = self.device)

    def reset_state(self):
        self.penalty = 0
        self.error = float('inf')

    def evaluate(self, inference):
        self.reset_state()
        values = self.process_inference(inference)
        self.apply(values)
        angles = self.get_joints()
        self.calc_error(angles)
        return self.error

    def process_inference(self, inference):
        values = [a.item() for a in inference]
        for idx, value in enumerate(values):
            name = self.joints[idx]
            limits = self.motion.getLimits(name)
            min_angle = limits[0][0]
            max_angle = limits[0][1]
            max_vel = limits[0][2]  # Unenforced
            max_tor = limits[0][3]  # Unenforced
            value = self.cap_angle(value, min_angle, max_angle)
            values[idx] = [value]
        return values

    def cap_angle(self, x, a, b):
        penalty = 10  # Safety penalty
        if x<a:
            self.penalty += penalty
            x = a
        elif x>b:
            self.penalty += penalty
            x = b
        return x

    def apply(self, angles):
        self.set_joints(angles)

    def calc_error(self, angles):
        errors = [abs(x-y) for x,y in zip(angles, self.target_angles)]
        self.error = sum(errors)
        self.error += self.penalty
        self.error = torch.tensor(self.error)

















#
