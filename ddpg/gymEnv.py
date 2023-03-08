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

    def __init__(self, render_mode="human", max_steps=500, initState=[-2, -2, 0, 0], goal=[2, 2]):
        self.window_size = 512
        self.field_limit = [-5.1, 5.1]
        self.successThreshold = 0.5
        self.factor = self.window_size / \
            (self.field_limit[1] - self.field_limit[0])

        self.initState = initState
        self.time = 0
        self.dt = 0.2
        self.old_angle = 0
        self.maxTime = max_steps*self.dt
        self.timePenalty = 0
        self.actionPenalty = 0
        self.distancePenalty = 0
        self.obstaclePenalty = -100
        self.reward_goal = 1000

        self.obstacle_color = (100, 0, 0)
        self.goal_color = (0, 200, 100)
        self.mouse_color = (80, 70, 255)
        self.background_color = (245, 240, 200)

        # self.A = np.array([[1, 0, self.dt, 0],
        #                   [0, 1, 0, self.dt],
        #                   [0, 0, 1, 0],
        #                   [0, 0, 0, 1]], dtype=np.float32)

        self.A = np.array([[1, 0],
                           [0, 1]], dtype=np.float32)

        # self.B = np.array([[0, 0],
        #                   [0, 0],
        #                   [self.dt, 0],
        #                   [0, self.dt]], dtype=np.float32)

        self.B = np.array([[self.dt, 0],
                           [0, self.dt]], dtype=np.float32)

        # self.state = {
        #     'x': initState[0],
        #     'y': initState[1],
        #     'vx': initState[2],
        #     'vy': initState[3]
        # }

        self.state = {
            'x': initState[0],
            'y': initState[1]
        }

        self.goal = {
            'x': goal[0],
            'y': goal[1]
        }
        self.max_steps = max_steps
        self.history = []

        self.actions = ["vertical_force", "horizontal_force"]

        # self.observations = ['x', 'y', 'vx', 'vy']
        self.observations = ['x', 'y']

        low = np.array(-1, dtype=np.float32)
        high = np.array(1, dtype=np.float32)
        self.action_space = spaces.Box(
            low=low, high=high)

        self.first_run = True
        self.obstacle_set = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.observation_space = self.make_mouse_obs_space()

        self.window = None
        self.clock = None

    def make_mouse_obs_space(self):
        # observations = ['x', 'y', 'vx', 'vy']
        observations = ['x', 'y']

        lower_obs_bound = {
            'x': - np.inf,
            'y': - np.inf,
            # 'vx': - np.inf,
            # 'vy': - np.inf
        }
        higher_obs_bound = {
            'x': np.inf,
            'y': np.inf,
            # 'vx': np.inf,
            # 'vy': np.inf
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
        self.history = []
        x0 = np.random.uniform(self.field_limit[0], self.field_limit[1])
        y0 = np.random.uniform(self.field_limit[0], self.field_limit[1])

        self.initState = np.array([x0, y0, 0, 0])
        # self.state = {
        #     'x': self.initState[0],
        #     'y': self.initState[1],
        #     'vx': self.initState[2],
        #     'vy': self.initState[3]
        # }
        self.state = {
            'x': self.initState[0],
            'y': self.initState[1]
        }

        self.time = 0

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        self.action = action
        state = self._get_obs()
        self.history.append(state)
        observation = np.matmul(self.A, np.squeeze(state)) +\
            np.matmul(self.B, np.squeeze(action))
        terminated = self.check_done()
        reward = self.get_reward()
        self.time += self.dt

        # self.state = {
        #     'x': observation[0],
        #     'y': observation[1],
        #     'vx': observation[2],
        #     'vy': observation[3],
        # }
        self.state = {
            'x': observation[0],
            'y': observation[1]
        }

        info = self._get_info()
        self.time += self.dt

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, info

    def get_reward(self):
        actionAmplitude = np.sqrt(self.action[0]**2 + self.action[1]**2)
        distFromGoal = np.sqrt(
            (self.state['x'] - self.goal['x'])**2 + (self.state['y'] - self.goal['y'])**2)

        if self.check_done():
            if distFromGoal <= self.successThreshold:
                return self.reward_goal
            elif self.time > self.maxTime:
                return self.distancePenalty*distFromGoal + self.actionPenalty * actionAmplitude
            else:
                return self.timePenalty + self.obstaclePenalty +\
                    self.distancePenalty*distFromGoal + self.actionPenalty * actionAmplitude
        else:
            return self.timePenalty + self.distancePenalty*distFromGoal + self.actionPenalty * actionAmplitude

    def check_done(self):
        if np.sqrt((self.state['x'] - self.goal['x'])**2 + (self.state['y'] - self.goal['y'])**2) <= self.successThreshold:
            return 1
        elif self.time > self.maxTime:
            return 1
        elif abs(self.state['x']) >= 5 or abs(self.state['y']) >= 5:
            return 1
        elif self.pos_to_pixel([self.state['x'], self.state['y']]) in self.obstacle_set:
            return 1
        else:
            return 0

    def pos_to_pixel(self, pos):
        return (int(pos[0] * self.factor + self.window_size / 2),
                -int(pos[1] * self.factor - self.window_size / 2))

    def render(self):
        if self.render_mode == "human":
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
        self.canvas = canvas
        canvas.fill(self.obstacle_color)

        pygame.draw.rect(canvas,
                         self.background_color,
                         pygame.Rect(
                             self.factor*0.1,
                             self.factor*0.1,
                             self.window_size - self.factor*0.2,
                             self.window_size - self.factor*0.2))

        # goal
        pygame.draw.circle(
            canvas,
            self.goal_color,
            self.pos_to_pixel([self.goal['x'], self.goal['y']]),
            int(self.factor * self.successThreshold))

        # initial state
        pygame.draw.circle(
            canvas,
            self.mouse_color,
            self.pos_to_pixel([self.initState[0], self.initState[1]]),
            int(self.factor * 0.2))

        # history of states
        for state in self.history:
            pygame.draw.circle(
                canvas,
                self.mouse_color,
                self.pos_to_pixel([state[0], state[1]]),
                int(self.factor * 0.05))

        self.multiple_small_polytope(canvas)
        self.plotMouse(canvas)

        if self.first_run:
            self.obstacle_set = self.get_obstacle_set()
            self.first_run = False

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

    def compute_mouse_points(self, angle, size=1):
        R = self.rotation_matrix(angle)

        vec_1 = size*np.array([1, 0])
        vec_2 = size*np.array([-0.5, 0.5])
        vec_3 = size*np.array([-0.5, -0.5])

        rot_point_1 = np.dot(R, vec_1)
        rot_point_2 = np.dot(R, vec_2)
        rot_point_3 = np.dot(R, vec_3)

        point_x = (self.state['x'] + rot_point_1[0],
                   self.state['y'] + rot_point_1[1])
        point_y = (self.state['x'] + rot_point_2[0],
                   self.state['y'] + rot_point_2[1])
        point_z = (self.state['x'] + rot_point_3[0],
                   self.state['y'] + rot_point_3[1])
        return point_x, point_y, point_z

    def plotMouse(self, canvas):
        # get tilt of triangle from velocity
        triangle_size = 0.5
        if len(self.history) == 0:
            new_angle = 0
        else:
            vx = self.state['x'] - self.history[-1][0]
            vy = self.state['y'] - self.history[-1][1]
            if vx == 0 and vy == 0:
                new_angle = 0
            else:
                new_angle = np.arctan2(vy, vx)
                if new_angle < 0:
                    new_angle += 2 * np.pi

        angle = (new_angle + self.old_angle)/2
        self.old_angle = angle
        point_x, point_y, point_z = self.compute_mouse_points(
            angle, triangle_size)
        point_x = self.pos_to_pixel(point_x)
        point_y = self.pos_to_pixel(point_y)
        point_z = self.pos_to_pixel(point_z)

        pygame.draw.polygon(
            canvas,
            self.mouse_color,
            [point_x, point_y, point_z]
        )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def rotation_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    def multiple_small_polytope(self, canvas):
        p1 = [0, 0]
        p2 = [-2, 2]
        p3 = [-1, 3]

        offset1 = [1, 1]
        offset2 = [3, 4]
        offset3 = [3, 3]

        angle1 = np.deg2rad(35)
        angle2 = np.deg2rad(75)
        angle3 = np.deg2rad(140)

        R1 = self.rotation_matrix(angle1)
        R2 = self.rotation_matrix(angle2)
        R3 = self.rotation_matrix(angle3)

        p1r1 = self.pos_to_pixel(np.dot(R1, p1) + offset1)
        p2r1 = self.pos_to_pixel(np.dot(R1, p2) + offset1)
        p3r1 = self.pos_to_pixel(np.dot(R1, p3) + offset1)

        p1r2 = self.pos_to_pixel(np.dot(R2, p1) + offset2)
        p2r2 = self.pos_to_pixel(np.dot(R2, p2) + offset2)
        p3r2 = self.pos_to_pixel(np.dot(R2, p3) + offset2)

        p1r3 = self.pos_to_pixel(np.dot(R3, p1) + offset3)
        p2r3 = self.pos_to_pixel(np.dot(R3, p2) + offset3)
        p3r3 = self.pos_to_pixel(np.dot(R3, p3) + offset3)

        pygame.draw.polygon(
            canvas,
            self.obstacle_color,
            [p1r1, p2r1, p3r1]
        )
        # pygame.draw.polygon(
        #     canvas,
        #     self.obstacle_color,
        #     [p1r2, p2r2, p3r2]
        # )
        pygame.draw.polygon(
            canvas,
            self.obstacle_color,
            [p1r3, p2r3, p3r3]
        )

    def bean_obstacle(self, canvas):
        """ Draws a bean obstacle"""
        p1 = (0, 0)
        p2 = (-2, 2)
        p3 = (-1, 3)
        p4 = (1, 2)
        p5 = (2, 0)
        p6 = (1, -2)
        p7 = (-1, -3)
        p8 = (-2, -2)
        offset = 80
        p1_out = (self.window_size/2 + 0.5*self.factor*p1[0] + offset,
                  self.window_size/2 + 0.5*self.factor*p1[1])
        p2_out = (self.window_size/2 + 0.5*self.factor*p2[0] + offset,
                  self.window_size/2 + 0.5*self.factor*p2[1])
        p3_out = (self.window_size/2 + 0.5*self.factor*p3[0] + offset,
                  self.window_size/2 + 0.5*self.factor*p3[1])
        p4_out = (self.window_size/2 + 0.5*self.factor*p4[0] + offset,
                  self.window_size/2 + 0.5*self.factor*p4[1])
        p5_out = (self.window_size/2 + 0.5*self.factor*p5[0] + offset,
                  self.window_size/2 + 0.5*self.factor*p5[1])
        p6_out = (self.window_size/2 + 0.5*self.factor*p6[0] + offset,
                  self.window_size/2 + 0.5*self.factor*p6[1])
        p7_out = (self.window_size/2 + 0.5*self.factor*p7[0] + offset,
                  self.window_size/2 + 0.5*self.factor*p7[1])
        p8_out = (self.window_size/2 + 0.5*self.factor*p8[0] + offset,
                  self.window_size/2 + 0.5*self.factor*p8[1])
        points_out = [p1_out, p2_out, p3_out, p4_out,
                      p5_out, p6_out, p7_out, p8_out]
        pygame.draw.polygon(
            canvas,
            self.obstacle_color,  # use black for the obstacle
            points_out
        )

    def get_obstacle_set(self):
        """ get the set of pixels which are colored by the obstacle method"""
        obstacle_set = []
        for pixel_x in range(self.window_size):
            for pixel_y in range(self.window_size):
                if self.canvas.get_at((pixel_x, pixel_y)) == self.obstacle_color:
                    obstacle_set.append((pixel_x, pixel_y))
        return obstacle_set

    def get_border_set(self):
        """ get the set of pixels which are on the frontier between obstacle and free environment"""
        border_set = []
        obstacle_set = self.get_obstacle_set()
        for pos_x in np.linspace(-5, 5, self.window_size):
            for pos_y in np.linspace(5, -5, self.window_size):
                pixel_x, pixel_y = self.pos_to_pixel((pos_x, pos_y))
                if (pixel_x, pixel_y) not in obstacle_set:
                    if (self.canvas.get_at((pixel_x - 1, pixel_y - 1)) == self.obstacle_color):
                        border_set.append((pos_x, pos_y))
                    elif (self.canvas.get_at((pixel_x, pixel_y - 1)) == self.obstacle_color):
                        border_set.append((pos_x, pos_y))
                    elif (self.canvas.get_at((pixel_x + 1, pixel_y - 1)) == self.obstacle_color):
                        border_set.append((pos_x, pos_y))
                    elif (self.canvas.get_at((pixel_x - 1, pixel_y)) == self.obstacle_color):
                        border_set.append((pos_x, pos_y))
                    elif (self.canvas.get_at((pixel_x + 1, pixel_y)) == self.obstacle_color):
                        border_set.append((pos_x, pos_y))
                    elif (self.canvas.get_at((pixel_x - 1, pixel_y + 1)) == self.obstacle_color):
                        border_set.append((pos_x, pos_y))
                    elif (self.canvas.get_at((pixel_x, pixel_y + 1)) == self.obstacle_color):
                        border_set.append((pos_x, pos_y))
                    elif (self.canvas.get_at((pixel_x + 1, pixel_y + 1)) == self.obstacle_color):
                        border_set.append((pos_x, pos_y))

        return border_set
