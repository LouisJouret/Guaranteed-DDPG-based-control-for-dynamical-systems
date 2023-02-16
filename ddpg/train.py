# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from agent import Agent
import utils
from gymEnv import Mouse
import gym

# env = gym.make('Pendulum-v0')
# agent = Agent(env.action_space.shape[0], env.observation_space.shape[0])
env = Mouse()
agent = Agent(len(env.actions), len(env.observations))
episodes = 1000
movAvglength = 100
episodeScore = []
episodeAvgScore = []
lastAvg = 0
best = -1000
testing = False

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

    if episode % 50 == 0:
        utils.plotQ(agent)
        utils.plotAction(agent)

utils.plotReward(episodeAvgScore)
