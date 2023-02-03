# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf


class doubleIntegrator():
    def __init__(self, x0, y0, goal_x, goal_y):
        self.x0 = x0
        self.y0 = y0
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.x = tf.constant(x0, dtype=tf.float32)
        self.y = tf.constant(y0, dtype=tf.float32)
        self.vx = tf.constant(0, dtype=tf.float32)
        self.vy = tf.constant(0, dtype=tf.float32)
        self.ux = tf.constant(0, dtype=tf.float32)
        self.uy = tf.constant(0, dtype=tf.float32)
        self.t = 0
        self.dt = 0.1
        self.maxTime = 5
        self.state = tf.constant(
            [[x0, y0, 0, 0]], dtype=tf.float32)
        self.A = tf.constant([[1, 0, self.dt, 0],
                              [0, 1, 0, self.dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=tf.float32)
        self.B = tf.constant([[0, 0],
                              [0, 0],
                              [self.dt, 0],
                              [0, self.dt]], dtype=tf.float32)
        self.successThreshold = 0.1

    def step(self, state, action):
        x_new = tf.linalg.matvec(self.A, state) + \
            tf.linalg.matvec(self.B, action)
        done = self.check_done()
        reward = self.get_reward()
        self.t += self.dt
        self.state = x_new
        return x_new, reward, done

    def reset(self):
        self.x = tf.constant(self.x0)
        self.y = tf.constant(self.y0)
        self.vx = 0
        self.vy = 0
        self.ux = 0
        self.uy = 0
        self.t = 0
        # self.render()

    def get_reward(self):
        reward = 0
        if abs(self.x - self.goal_x) < self.successThreshold and abs(self.y - self.goal_y) < self.successThreshold:
            if self.t == 0:
                return reward
            reward += 100
            reward += 20*(self.maxTime / self.t)
        elif self.t >= self.maxTime:
            reward -= (self.x - self.goal_x)**2 + (self.y - self.goal_y)**2
        return tf.constant(reward, dtype=tf.float32)

    def check_done(self):

        if abs(self.x - self.goal_x) < self.successThreshold and abs(self.y - self.goal_y) < self.successThreshold:
            return tf.constant(1, dtype=tf.float32)
        elif self.t > self.maxTime:
            return tf.constant(1, dtype=tf.float32)
        else:
            return tf.constant(0, dtype=tf.float32)

    def render(self):
        fig, ax = plt.subplots()
        ax.set_xlim(left=-5, right=5)
        ax.set_ylim(bottom=-5, top=5)
        ax.set_aspect('equal', adjustable='box')
        speed_up = 2
        FRAMES_NUMBER = round(len(self.buffer)/speed_up)

        def animate(frame):
            if frame == FRAMES_NUMBER-1:
                plt.close()
            else:
                ax.plot(self.buffer[speed_up*frame][0],
                        self.buffer[speed_up*frame][1], 'ro')

        goal = plt.Circle((self.goal_x, self.goal_y),
                          self.successThreshold, color='blue')
        ax.add_patch(goal)
        anim = animation.FuncAnimation(
            fig, animate, frames=round(len(self.Rbuffer)/speed_up), repeat=False)
        plt.show()
