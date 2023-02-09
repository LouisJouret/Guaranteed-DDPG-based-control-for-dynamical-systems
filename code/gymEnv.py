# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import gym
from gym import spaces
import pygame
import numpy as np
from dynamicModel import doubleIntegrator


class Mouse(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, max_steps=1000, initState=[0, 0, 0, 0], goal=[2, 2]):
        self.window_size = 512  # The size of the PyGame window
        self.mouse = doubleIntegrator(
            initState[0], initState[1], goal[0], goal[1])

        self.initState = initState

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
        low = np.array(-1)
        high = np.array(1)
        self.action_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

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
        self.steps_left = self.max_steps

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        state = self._get_obs()
        observation, reward, terminated = self.mouse.step(state, action)
        self.state = {
            'x': observation[0],
            'y': observation[1],
            'vx': observation[2],
            'vy': observation[3],
        }
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        factor = self.window_size / 10

        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (int(factor * self.goal['x']), int(factor * self.goal['y'])),
            int(factor * 0.5)
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (int(factor * self.state['x']), int(factor * self.state['y'])),
            int(factor * 0.2)
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
