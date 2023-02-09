# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import gym
from gym import spaces
import pygame
import numpy as np


class Mouse(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode="human", max_steps=500, initState=[0, 0, 0, 0], goal=[2, 2]):
        self.window_size = 512
        self.field_limit = [-5, 5]
        self.successThreshold = 0.5
        self.factor = self.window_size / \
            (self.field_limit[1] - self.field_limit[0])

        # self.mouse = doubleIntegrator(
        #     initState[0], initState[1], goal[0], goal[1])

        self.initState = initState
        self.time = 0
        self.dt = 0.05
        self.maxTime = max_steps*self.dt

        self.A = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        self.B = np.array([[0, 0],
                          [0, 0],
                          [self.dt, 0],
                          [0, self.dt]], dtype=np.float32)

        self.state = {
            'x': initState[0],
            'y': initState[1],
            'vx': initState[2],
            'vy': initState[3]
        }
        self.goal = {
            'x': goal[0],
            'y': goal[1]
        }
        self.max_steps = max_steps

        self.actions = ["vertical_force", "horizontal_force"]

        self.observations = ['x', 'y', 'vx', 'vy']

        low = np.array(-1, dtype=np.float32)
        high = np.array(1, dtype=np.float32)
        self.action_space = spaces.Box(
            low=low, high=high)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.observation_space = self.make_mouse_obs_space()

        self.window = None
        self.clock = None

    def make_mouse_obs_space(self):
        observations = ['x', 'y', 'vx', 'vy']

        lower_obs_bound = {
            'x': - np.inf,
            'y': - np.inf,
            'vx': - np.inf,
            'vy': - np.inf
        }
        higher_obs_bound = {
            'x': np.inf,
            'y': np.inf,
            'vx': np.inf,
            'vy': np.inf
        }

        low = np.array([lower_obs_bound[obs] for obs in observations])
        high = np.array([higher_obs_bound[obs] for obs in observations])
        shape = (len(observations),)
        return gym.spaces.Box(low, high, shape)

    def _get_obs(self):
        return np.array([self.state[obs] for obs in self.observations])

    def _get_info(self):
        return {"distance": np.sqrt((self.state['x'] - self.goal['x']) ** 2 + (self.state['y'] - self.goal['y']) ** 2)}

    def reset(self):
        # super().reset(seed=seed)

        self.state = {
            'x': self.initState[0],
            'y': self.initState[1],
            'vx': self.initState[2],
            'vy': self.initState[3]
        }
        self.time = 0

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        state = self._get_obs()
        observation = np.matmul(self.A, np.squeeze(state)) +\
            np.matmul(self.B, np.squeeze(action))
        terminated = self.check_done()
        reward = self.get_reward()
        self.time += self.dt

        self.state = {
            'x': observation[0],
            'y': observation[1],
            'vx': observation[2],
            'vy': observation[3],
        }

        info = self._get_info()
        self.time += self.dt

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def get_reward(self):
        if self.check_done():
            if np.sqrt((self.state['x'] - self.goal['x'])**2 + (self.state['y'] - self.goal['y'])**2) <= self.successThreshold:
                if self.time == 0:
                    return 0
                else:
                    return 100
            elif abs(self.state['x']) > 5 or abs(self.state['y']) > 5:
                return -100
            else:
                return -1
        else:
            return -1

    def check_done(self):
        if np.sqrt((self.state['x'] - self.goal['x'])**2 + (self.state['y'] - self.goal['y'])**2) <= self.successThreshold:
            return 1
        elif self.time > self.maxTime:
            return 1
        elif abs(self.state['x']) > 5 or abs(self.state['y']) > 5:
            return 1
        else:
            return 0

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('MouseCheese')
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((243, 237, 202))

        pygame.draw.circle(
            canvas,
            (145, 22, 253),
            (int(self.factor * self.goal['x'] + self.window_size/2),
             int(self.factor * self.goal['y'] + self.window_size/2)),
            int(self.factor * self.successThreshold)
        )

        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (int(factor * self.state['x'] + self.window_size/2),
        #      int(factor * self.state['y'] + self.window_size/2)),
        #     int(factor * 0.2)
        # )

        # get tilt of triangle from velocity
        triangle_size = 0.5
        vx = self.state['vx']
        vy = self.state['vy']
        if vx == 0 and vy == 0:
            angle = 0
        else:
            angle = np.arctan2(vy, vx)
            if angle < 0:
                angle += 2 * np.pi

        point_x, point_y, point_z = self.compute_triangle(angle, triangle_size)
        pygame.draw.polygon(
            canvas,
            (134, 16, 3),
            [point_x, point_y, point_z]
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def compute_triangle(self, angle, size=1):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        vec_1 = size*np.array([1, 0])
        vec_2 = size*np.array([-0.5, 0.5])
        vec_3 = size*np.array([-0.5, -0.5])

        rot_point_1 = np.dot(R, vec_1)
        rot_point_2 = np.dot(R, vec_2)
        rot_point_3 = np.dot(R, vec_3)

        point_x = (int(self.window_size/2 + self.factor*(self.state['x'] + rot_point_1[0])),
                   int(self.window_size/2 + self.factor*(self.state['y'] + rot_point_1[1])))
        point_y = (int(self.window_size/2 + self.factor*(self.state['x'] + rot_point_2[0])),
                   int(self.window_size/2 + self.factor*(self.state['y'] + rot_point_2[1])))
        point_z = (int(self.window_size/2 + self.factor*(self.state['x'] + rot_point_3[0])),
                   int(self.window_size/2 + self.factor*(self.state['y'] + rot_point_3[1])))
        return point_x, point_y, point_z

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
