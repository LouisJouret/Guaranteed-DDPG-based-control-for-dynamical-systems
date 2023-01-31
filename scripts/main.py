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
import time
# from tensorflow.python.keras.optimizers import Adam

model = doubleIntegrator(-3, -3, 2, 2)

episodes = np.linspace(1, 2, 2)
for episode in episodes:
    state = model.get_state()
    done = False
    score = 0
    while not done:
        action = [-0.3*model.x, -0.1*model.y]
        n_state, reward, done, info = model.step(action)
        score += reward
    model.reset()
    print('episode {} score {}'.format(episode, score))
