# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from dynamicModel import doubleIntegrator
import numpy as np
from agent import Agent
import tensorflow as tf
from utils import plotQ, plotReward


def main() -> None:

    tf.random.set_seed(165835)
    agent = Agent()
    episodes = 5
    movAvglength = 10
    episodeScore = [0]*movAvglength
    episodeAvgScore = []
    lastAvg = 0

    for episode in range(episodes):
        x0 = random.randint(-5, 5)
        y0 = random.randint(-5, 5)
        system = doubleIntegrator(x0, y0, 0, 0)
        done = False
        score = 0
        while not done:
            state = system.state
            action = agent.act(state)
            nextState, reward, done = system.step(state, action)
            agent.replayBuffer.storexp(
                state, action, reward, done, nextState)
            agent.train()
            score += reward
        episodeScore.append(score)
        lastAvg = np.mean(episodeScore[-movAvglength:])
        episodeAvgScore.append(lastAvg)
        system.reset()
        print(
            f"total reward after {episode} steps is {score} and last {movAvglength} episode average is {lastAvg}")

    plotReward(episodeAvgScore)
    plotQ(agent, tf.constant([[1., 1.]], dtype=tf.float32))


if __name__ == "__main__":
    main()
