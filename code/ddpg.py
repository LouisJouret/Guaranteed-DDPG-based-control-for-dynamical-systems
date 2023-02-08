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
from test_models import test


# tf.random.set_seed(165835)
# env = gym.make('')
system = doubleIntegrator(-2, -3, 4.5, 4)
agent = Agent()
episodes = 300
movAvglength = 50
episodeScore = []
episodeAvgScore = []
lastAvg = 0
best = -1000
testing = True

if testing:
    test(agent)
else:
    for episode in range(episodes):
        done = False
        score = 0
        system.reset()

        while not done:
            observation = system.state
            action = agent.act(observation)
            nextObservation, reward, done = system.step(
                observation, np.array(action))
            agent.replayBuffer.storexp(
                observation, action, reward, done, nextObservation)
            agent.train()
            score += reward
        episodeScore.append(score)
        if episode > movAvglength:
            avg = np.mean(episodeScore[-movAvglength:])
        else:
            avg = np.mean(episodeScore)
        print(
            f" episode {episode} has a score of {score} and an average score of {avg}")
        episodeAvgScore.append(avg)
        if best < avg:
            agent.save()
            best = avg

    utils.plotReward(episodeAvgScore)
