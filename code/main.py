# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from dynamicModel import doubleIntegrator
import numpy as np
from agent import Agent
import tensorflow as tf
import utils


def main() -> None:

    # tf.random.set_seed(165835)
    agent = Agent()
    episodes = 100
    movAvglength = 50
    episodeScore = [0]*movAvglength
    episodeAvgScore = []
    lastAvg = 0

    for episode in range(episodes):
        x0, y0 = utils.getInitialPoint()
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
        # for _ in range(agent.maxBufferSize):
        #     agent.train()

        episodeScore.append(score)
        lastAvg = np.mean(episodeScore[-movAvglength:])
        episodeAvgScore.append(lastAvg)
        system.reset()

        # print(agent.actorMain.get_weights()[0])
        print(
            f"total reward after {episode} steps is {score}")

    utils.plotReward(episodeAvgScore)
    utils.plotQ(agent, tf.constant([[1, 0]], dtype=tf.float32))


if __name__ == "__main__":
    main()
