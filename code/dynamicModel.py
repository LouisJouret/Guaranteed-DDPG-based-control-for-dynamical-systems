# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math


class doubleIntegrator():
    def __init__(self, x0, y0, goal_x, goal_y):
        self.x0 = x0
        self.y0 = y0
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.t = 0
        self.dt = 0.1
        self.maxTime = 10
        self.state = np.array([[x0, y0, 0, 0]], dtype=np.float32)

        self.A = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        self.B = np.array([[0, 0],
                          [0, 0],
                          [self.dt, 0],
                          [0, self.dt]], dtype=np.float32)
        self.successThreshold = 1.0

    def step(self, state, action):
        x_new = np.matmul(self.A, np.squeeze(state)) +\
            np.matmul(self.B, np.squeeze(action))
        done = self.check_done()
        reward = self.get_reward()
        self.t += self.dt
        self.state = np.array([x_new], dtype=np.float32)

        return x_new, reward, done

    def reset(self):
        self.state = np.array([[self.x0, self.y0, 0, 0]], dtype=np.float32)
        self.t = 0
        # self.render()

    def get_reward(self):
        if self.check_done():
            if math.sqrt((self.state[0][0] - self.goal_x)**2 + (self.state[0][1] - self.goal_y)**2) < self.successThreshold:
                if self.t == 0:
                    return 0
                else:
                    return 100
            elif abs(self.state[0][0]) > 5 or abs(self.state[0][1]) > 5:
                return -100
        return 0

    def check_done(self):

        if abs(self.state[0][0] - self.goal_x) < self.successThreshold and abs(self.state[0][1] - self.goal_y) < self.successThreshold:
            return 1
        elif self.t > self.maxTime:
            return 1
        elif abs(self.state[0][0]) > 5 or abs(self.state[0][1]) > 5:
            return 1
        else:
            return 0
