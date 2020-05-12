import numpy as np
import math
# For comparison
np.random.seed(17080399)  # good for scenario 2 & 3
# np.random.seed(9042020)     # good for scenario 1


class usv:
    # update rate (assumed to be same for every sensor)
    dt = 1
    # body frame maximum acceleration
    MAX_AX = 1
    MAX_AY = 1
    # body frame maximum velocity
    MAX_VX = 2
    MAX_VY = 10
    # body frame maximum turning rate math.radians(deg/s)
    MAX_W = math.radians(20)

    gps_sd = 8              # meters
    bearing_sd = math.radians(1)
    imu_a_sd = 0.0042       # ms-2
    imu_w_sd = math.radians(0.1)        #rad/s
    imu_a_bias = 0.02       # ms-2
    imu_w_bias = math.radians(0.28)       #rad/s

    # heading takes in 0-359 degrees with 0 at North
    def __init__(self, po, vo, heading, the_max_w = math.radians(20)):
        # earth frame position
        self.pox = po[0]
        self.poy = po[1]
        # body frame velocity
        self.vox = vo[0]
        self.voy = vo[1]
        self.prev_vox = vo[0]
        self.prev_voy = vo[1]
        # navigation frame heading
        self.heading = math.radians(heading)
        self.prev_heading = self.heading
        self.target_head = self.heading
        self.head_step = 0
        # turning rate omega rad/s
        self.max_w = the_max_w
        self.omega = (self.heading - self.prev_heading) / self.dt
        # body frame forward velocity target
        self.target_v = self.voy
        self.v_step = 0
        # body frame acceleration
        self.abx = (self.vox - self.prev_vox) / self.dt
        self.aby = (self.voy - self.prev_voy) / self.dt

    def move(self):
        # record previous velocity and heading before any updates
        self.prev_vox = self.vox
        self.prev_voy = self.voy
        self.prev_heading = self.heading

        if self.head_step == 0:
            self.heading = self.target_head
        elif self.head_step < 0:
            self.heading -= self.max_w * self.dt
            self.head_step += 1
        elif self.head_step > 0:
            self.heading += self.max_w * self.dt
            self.head_step -= 1

        if self.v_step == 0:
            self.voy = self.target_v
        elif self.v_step < 0:
            self.voy -= self.MAX_AY * self.dt
            self.v_step += 1
        elif self.v_step > 0:
            self.voy += self.MAX_AY * self.dt
            self.v_step -= 1
        
        # if self.heading < 0:
        #     self.heading = 360 + self.heading
        # represent this into matrix? (TO DO)
        self.omega = (self.heading - self.prev_heading) / self.dt
        self.abx = (self.vox - self.prev_vox) / self.dt + (self.omega * self.voy)
        self.aby = (self.voy - self.prev_voy) / self.dt
        self.pox += (math.sin(self.heading)*self.voy + math.cos(self.heading)*self.vox) * self.dt
        self.poy += (math.cos(self.heading)*self.voy - math.sin(self.heading)*self.vox) * self.dt

        # ADD NOISE TO REPRESENT SURFACE DISRUPTIONS AND RECALCULATE ATTRIBUTES

    @property
    def bearing(self):
        reading = self.bearing_sd * np.random.randn() + self.heading
        return reading

    @property
    def gps(self):
        gpsx = self.gps_sd * np.random.randn() + self.pox
        gpsy = self.gps_sd * np.random.randn() + self.poy
        return [gpsx, gpsy]

    @property
    def imu(self):
        # SD + Mean + Bias
        imu_abx = self.imu_a_sd * np.random.randn() + self.abx + self.imu_a_bias
        imu_aby = self.imu_a_sd * np.random.randn() + self.aby + self.imu_a_bias
        imu_w = self.imu_w_sd * np.random.randn() + self.omega + self.imu_w_bias
        return [imu_abx, imu_aby, imu_w]

    # command: -n degrees or n degrees, turning_rate in deg/s
    def cmd_heading(self, command, turning_rate):
        self.target_head = self.heading + math.radians(command)
        rate = math.radians(command) / (math.radians(turning_rate) * self.dt)
        self.max_w = math.radians(turning_rate)
        # 0 step needed if within turning rate (immediate change to target)
        # negative if left turn, positive if right turn
        self.head_step = int(rate)

    # command: -n m/s or n m/s
    def cmd_vel(self, command):
        # validate speed input
        if abs(command) + self.voy > self.MAX_VY:
            command = (self.MAX_VY - self.voy) * (command / abs(command))
        elif abs(command) + self.voy < 0:
            command = -self.voy
        self.target_v += command
        rate = command / (self.MAX_AY * self.dt)
        self.v_step = int(rate)