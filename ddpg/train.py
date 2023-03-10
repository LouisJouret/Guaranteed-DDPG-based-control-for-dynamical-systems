# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from agent import Agent
import utils
from gymEnv import Mouse

env = Mouse()
agent = Agent(len(env.actions), len(env.observations))
# run the model with a dummy input to initialize the weights
dummy_state = np.zeros((1, len(env.observations)))
agent.actorMain(dummy_state)
episodes = 2000
movAvglength = 100
episodeScore = []
episodeAvgScore = []
lastAvg = 0
best = -100
history_succes = []
percent_succes = 0

for episode in range(episodes):
    done = False
    score = 0
    observation = env.reset()
    while not done:
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

    if episode % 100 == 2:
        utils.plotActionVectors(agent, env, episode)
        utils.plotActionBorderVectors(agent, env, episode)
        utils.plotAction(agent, episode)
        utils.plotQ(agent, episode)
        # utils.plotLinearRegion(agent, episode)


utils.plotReward(episodeAvgScore)
