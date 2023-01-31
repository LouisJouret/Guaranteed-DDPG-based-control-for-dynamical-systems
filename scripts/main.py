# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
from model import doubleIntegrator
# from tensorflow.python.keras.optimizers import Adam

# env = gym.make('CartPole-v1')
model = doubleIntegrator(-1, -1)
# actions = env.action_space.n
# print('Actions', actions)

episodes = np.linspace(1, 11)
for episode in episodes:
    state = model.get_state()
    done = False
    score = 0
    while not done:
        action = [1, 1]
        model.render()
        n_state, reward, done, info = model.step(action)
        score += reward

    print('episode {} score {}'.format(episode, score))
