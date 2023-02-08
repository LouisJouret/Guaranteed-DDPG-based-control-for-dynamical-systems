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


# tf.random.set_seed(165835)
env = gym.make('Pendulum-v0')
agent = Agent()
episodes = 200
movAvglength = 50
episodeScore = [0]*movAvglength
episodeAvgScore = []
lastAvg = 0


for episode in range(episodes):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        env.render()
        action = agent.act(np.array([observation]))
        nextObservation, reward, done, _ = env.step(action[0])
        agent.replayBuffer.storexp(
            observation, action, reward, done, nextObservation)
        agent.train()
        observation = nextObservation
        score += reward
    episodeScore.append(score)
    print(
        f" episode {episode} has a score of {score} and an average score of {np.mean(episodeScore[-10:])}")
    lastAvg = np.mean(episodeScore[-movAvglength:])
    episodeAvgScore.append(lastAvg)


utils.plotReward(episodeAvgScore)
