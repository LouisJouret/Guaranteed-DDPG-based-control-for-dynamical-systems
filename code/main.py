# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from agent import Agent
import numpy as np
from dynamicModel import doubleIntegrator
import random
from matplotlib.animation import FuncAnimation
from functools import partial
import time
from matplotlib import pyplot as plt
from datetime import datetime


def main() -> None:

    with tf.device('GPU:0'):
        tf.random.set_seed(165835)
        agent = Agent()
        episodes = 100
        episodeScore = [0]*5
        lastAvg = 0

        for episode in range(episodes):
            x0 = random.randint(-3, 3)
            y0 = random.randint(-3, 3)
            system = doubleIntegrator(x0, y0, 0, 0)
            done = False
            score = 0
            while not done:
                state = tf.constant(system.get_state(), dtype=tf.float32)
                action = agent.act(state)
                nextState, reward, done, _ = system.step(action)
                nextState = tf.constant(nextState, dtype=tf.float32)
                reward = tf.constant(reward, dtype=tf.float32)
                done = tf.constant(done, dtype=tf.float32)
                values = (state, action, reward, done, nextState)
                values_batched = tf.nest.map_structure(
                    lambda t: tf.stack([t] * agent.batchSize), values)
                agent.replayBuffer.add_batch(values_batched)
                agent.train()
                state = nextState
                score += reward
            episodeScore.append(score)
            lastAvg = np.mean(episodeScore[-5:])
            system.reset()
            print(
                f"total reward after {episode} steps is {score} and episode average is {lastAvg}")


if __name__ == "__main__":
    main()
