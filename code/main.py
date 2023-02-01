# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from agent import Agent
import numpy as np
from dynamicModel import doubleIntegrator


system = doubleIntegrator(-3, -3, 1, 1)


with tf.device('GPU:0'):
    tf.random.set_seed(165835)
    agent = Agent()
    episodes = 10
    episodesPerformance = []

    for episode in episodes:
        state = system.reset()
        done = False
        score = 0
        while not done:
            action = agent.act(state)
            statesNext, reward, done, _ = system.step(action)
            agent.buffer.append([statesNext, reward])
            agent.train()
            state = statesNext
            score += reward
        system.reset()
        episodesPerformance.append(score)
        movAvgPerf = np.mean(episodesPerformance[-100:])
        print(
            f"total reward after {episode} steps is {score} and avg reward is {movAvgPerf[-1]}")
