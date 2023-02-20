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
history_succes = []
percent_succes = 0

for episode in range(episodes):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        # env.render()
        action = agent.act(np.array([observation]))
        nextObservation, reward, done, _ = env.step(action[0])
        agent.replayBuffer.storexp(
            observation, action, reward, done, nextObservation)
        agent.train()
        observation = nextObservation
        score += reward
    if reward > 0:
        history_succes.append(1)
    else:
        history_succes.append(0)
    episodeScore.append(score)
    if episode > movAvglength:
        avg = np.mean(episodeScore[-movAvglength:])
        percent_succes = np.sum(
            history_succes[-movAvglength:])*100/movAvglength
    else:
        avg = np.mean(episodeScore)
        percent_succes = np.sum(
            history_succes)*100/movAvglength
    print(
        f" episode {episode} has a score of {score} and an average success of {percent_succes} %")
    episodeAvgScore.append(avg)
    if best < avg:
        agent.save()
        best = avg

    if episode % 100 == 0:
        utils.plotActionVectors(agent, env, episode)
        utils.plotAction(agent, episode)

utils.plotReward(episodeAvgScore)
