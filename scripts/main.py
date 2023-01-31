import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
import time
# from tensorflow.python.keras.optimizers import Adam

env = gym.make('CartPole-v1')

actions = env.action_space.n
print('Actions', actions)

episodes = np.linspace(1, 11)
for episode in episodes:
    # At each begining reset the game
    state = env.reset()
    # set done to False
    done = False
    # set score to 0
    score = 0
    # while the game is not finished
    while not done:
        # visualize each step
        # choose a random action
        action = random.choice([0, 1])
        env.render()
        # execute the action
        n_state, reward, done, info = env.step(action)

        # keep track of rewards
        score += reward
        # set done to True when the game is terminated

    print('episode {} score {}'.format(episode, score))
