"""NAO robot class"""

from ..environment import Environment
from PIL import Image
import time
#import qi
#from naoqi import ALProxy

class Robot(Environment):
    def __init__(self, env_params):
        super(Robot, self).__init__(env_params)
        env_params = self.ingest_params_lvl1(env_params)
        self.robot = True
        self.ip = env_params["ip"]
        self.port = env_params["port"]
        self.tts = None
        self.motion = None
        self.posture = None
        self.sensors = None
        self.init_robot()
        self.minimize = True

    def ingest_params_lvl1(self, env_params):
        assert type(env_params) is dict
        if "ip" not in env_params:
            env_params["ip"] = "localhost"
        if "port" not in env_params:
            env_params["port"] = 58463
        if "score type" not in env_params:
            env_params["score type"] = "score"
        return env_params

    def init_robot(self):
        url = "tcp://" + self.ip + ":" + str(self.port)
        app = qi.Application(["Research", "--qi-url=" + url])
        app.start()
        self.session = app.session
        self.motion = self.session.service("ALMotion")
        self.posture = self.session.service("ALRobotPosture")
        self.tts = self.session.service("ALTextToSpeech")
        self.memory = self.session.service("ALMemory")
        self.video = self.session.service("ALVideoDevice")
        self.subscribe_camera()

    def subscribe_camera(self):
        resolution = 2  # VGA
        color_space = 11  # RGB
        fps = 5
        self.camera = self.video.subscribe("python_client", resolution, color_space, fps)

    def say(self, message):
        self.tts.say(message)

    def assume_pose(self, pose):
        self.motion.wakeUp()
        self.posture.goToPosture(pose, 0.5)

    def rest(self):
        self.motion.rest()

    def get_joints(self):
        useSensors = False
        state = []
        for joint in self.joints:
            angle = self.motion.getAngles(joint, useSensors)
            state.extend(angle)
        return state

    def set_joints(self, values):
        print("Setting NAO to", values)
        x = len(self.joints)
        print(x)
        print(len(values))
        assert x == len(values)  # Sanity check
        times = [[3.7]]*x  # seconds
        self.motion.angleInterpolationBezier(self.joints, times, values)

    def set_joints_(self, values):
        print("Setting NAO to", values)
        x = len(self.joints)
        assert x == len(values)  # Sanity check
        speed_fraction = 0.15
        self.motion.setAngles(self.joints, values, speed_fraction)
        time.sleep(1.5)

    def get_sensors(self, names):
        names = self.set_sensor_names(names)
        readings = []
        for sensor in names:
            reading = self.memory.getData(sensor)
            readings.extend(reading)
        return readings

    def set_sensor_names(self, names):
        if len(names) == 0:
            print("Can't fetch sensors without identification, empty list")
            exit()
        if names == 'Acc':
            x = "Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value"
            y = "Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value"
            z = "Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value"
            names = [x, y, z]

        elif names == 'Gyr':
            x = "Device/SubDeviceList/InertialSensor/GyroscopeX/Sensor/Value"
            y = "Device/SubDeviceList/InertialSensor/GyroscopeY/Sensor/Value"
            names = [x, y]

        elif names == 'Torso':
            x = "Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value"
            y = "Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value"
            names = [x, y]

        elif names == 'All':
            accx = "Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value"
            accy = "Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value"
            accz = "Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value"
            gyrx = "Device/SubDeviceList/InertialSensor/GyroscopeX/Sensor/Value"
            gyry = "Device/SubDeviceList/InertialSensor/GyroscopeY/Sensor/Value"
            angx = "Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value"
            angy = "Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value"
            names = [accx, accy, accz, gyrx, gyry, angx, angy]
        return names

    def get_image(self):
        image = self.video.getImageRemote(self.camera)
        #self.video.unsubscribe(self.camera)
        return image

    def save_image(self, image, name="nao_image", show=False):
        width = image[0]
        height = image[1]
        array = image[6]
        string = str(bytearray(array))
        im = Image.fromstring("RGB", (width, height), string)
        name = name+".png"
        im.save(name, "PNG")
        if show:
            im.show()








#
