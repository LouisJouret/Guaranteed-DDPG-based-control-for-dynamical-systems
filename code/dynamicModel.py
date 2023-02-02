# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


class doubleIntegrator():
    def __init__(self, x0, y0, goal_x, goal_y):
        self.x0 = x0
        self.y0 = y0
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.x = x0
        self.y = y0
        self.vx = 0
        self.vy = 0
        self.ux = 0
        self.uy = 0
        self.t = 0
        self.dt = 0.01
        self.buffer = []
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[0, 0],
                           [0, 0],
                           [self.dt, 0],
                           [0, self.dt]])

    def step(self, action):
        ux, uy = action
        x_new = np.matmul(self.A, np.array(
            [self.x, self.y, self.vx, self.vy])) + np.matmul(self.B, np.array([ux, uy]))
        self.x = x_new[0]
        self.y = x_new[1]
        self.vx = x_new[2]
        self.vy = x_new[3]
        done = self.check_done()
        reward = self.get_reward()
        self.buffer.append([self.x, self.y, self.vx, self.vy])

        return x_new, reward, done, {}

    def reset(self):
        self.x = self.x0
        self.y = self.y0
        self.vx = 0
        self.vy = 0
        self.ux = 0
        self.uy = 0
        self.t = 0
        self.render()
        self.buffer = []

    def get_state(self):
        return [self.x, self.y, self.vx, self.vy]

    def get_time(self):
        return self.t

    def get_position(self):
        return [self.x, self.y]

    def get_velocity(self):
        return [self.vx, self.vy]

    def get_time_step(self):
        return self.dt

    def get_state_size(self):
        return 4

    def get_action_size(self):
        return 2

    def get_reward(self):
        reward = 0
        threshold = 0.1
        if abs(self.x - self.goal_x) < threshold and abs(self.y - self.goal_y) < threshold:
            reward += 100
        reward -= self.t
        return reward

    def check_done(self):
        threshold = 0.1
        if abs(self.x - self.goal_x) < threshold and abs(self.y - self.goal_y) < threshold:
            return True
        else:
            return False

    def render(self):
        fig, ax = plt.subplots()
        ax.set_xlim(left=-5, right=5)
        ax.set_ylim(bottom=-5, top=5)
        ax.set_aspect('equal', adjustable='box')
        speed_up = 30

        def animate(idx):
            print(
                f"x = {round(self.buffer[speed_up*idx][0],2)} , y = {round(self.buffer[speed_up*idx][1],2)}")
            ax.plot(self.buffer[speed_up*idx][0],
                    self.buffer[speed_up*idx][1], 'ro')

        anim = animation.FuncAnimation(
            fig, animate, frames=round(len(self.buffer)/speed_up), repeat=False)
        plt.show()
        fig.clear()
        ax.clear()
