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
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = random.choice([0, 1])
        env.render()
        n_state, reward, done, info = env.step(action)
        score += reward

    print('episode {} score {}'.format(episode, score))
