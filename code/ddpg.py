# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from dynamicModel import doubleIntegrator
import numpy as np
from agent import Agent
import tensorflow as tf
import utils
import gym


tf.random.set_seed(165835)
env = gym.make('MountainCar-v0')
agent = Agent()
episodes = 200
movAvglength = 50
episodeScore = [0]*movAvglength
episodeAvgScore = []
lastAvg = 0

for episode in range(episodes):
    # x0, y0 = utils.getInitialPoint()
    # system = doubleIntegrator(x0, y0, 0, 0)
    done = False
    score = 0
    observation = env.reset()
    while not done:
        env.render()
        # state = system.state
        action = agent.act(observation)
        action = tf.cast(action, dtype=tf.int32)
        # action = env.action_space.sample()
        # print(round(action))
        # nextState, reward, done = system.step(state, action)
        nextObservation, reward, done, _ = env.step(np.array(action))
        # print(observation)
        # agent.replayBuffer.storexp(
        #     state, action, reward, done, nextState)
        agent.replayBuffer.storexp(
            observation, action, reward, done, nextObservation)

        agent.train()
        observation = nextObservation
        score += reward

        # print(agent.actorMain.get_weights()[0][0][0])
    episodeScore.append(score)
    lastAvg = np.mean(episodeScore[-movAvglength:])
    episodeAvgScore.append(lastAvg)
    # print(
    #     f"total reward after {episode} steps is {score} and the final state is ({system.state[0][0]},{system.state[0][1]})")
    # system.reset()

utils.plotReward(episodeAvgScore)
# utils.plotQ(agent, tf.constant([[1, 0]], dtype=tf.float32))
# utils.plotAction(agent)
