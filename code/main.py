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
from matplotlib import pyplot as plt


def main() -> None:

    with tf.device('GPU:0'):
        tf.random.set_seed(165835)
        agent = Agent()
        episodes = 100
        movAvglength = 10
        episodeScore = [0]*movAvglength
        lastAvg = 0

        for episode in range(episodes):
            x0 = random.randint(-3, 3)
            y0 = random.randint(-3, 3)
            system = doubleIntegrator(x0, y0, 0, 0)
            done = False
            score = 0
            while not done:
                state = system.get_state()
                action = agent.act(state)
                nextState, reward, done = system.step(state, action)
                agent.replayBuffer.storexp(
                    state, action, reward, done, nextState)
                agent.train()
                state = nextState
                score += reward
            episodeScore.append(score)
            lastAvg = np.mean(episodeScore[-movAvglength:])
            system.reset()
            print(
                f"total reward after {episode} steps is {score} and episode average is {lastAvg}")


if __name__ == "__main__":
    main()
