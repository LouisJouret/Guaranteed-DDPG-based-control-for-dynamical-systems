# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from agent import Agent
import numpy as np
from dynamicModel import doubleIntegrator


def main() -> None:
    system = doubleIntegrator(-3, -3, 1, 1)

    with tf.device('GPU:0'):
        tf.random.set_seed(165835)
        agent = Agent()
        episodes = 10
        episodesPerformance = []

        for episode in range(episodes):
            done = False
            score = 0
            while not done:
                state = tf.constant(system.get_state(), dtype=tf.float32)
                action = agent.act(state)
                nextState, reward, done, _ = system.step(action)
                nextState = tf.constant(nextState)
                reward = tf.constant(reward)
                done = tf.constant(done)
                values = (state, action, reward, done, nextState)
                agent.replayBuffer.add_batch(values)
                agent.train()
                state = nextState
                score += reward
            system.reset()
            episodesPerformance.append(score)
            movAvgPerf = np.mean(episodesPerformance[-100:])
            print(
                f"total reward after {episode} steps is {score} and avg reward is {movAvgPerf[-1]}")


if __name__ == "__main__":
    main()
