# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
from agent import Agent
import tensorflow as tf
import numpy as np


def plotQ(agent: Agent, action) -> None:
    "plots a 2D canvas of the Q-function for a given state and action"
    size = 50
    QArray = np.zeros((size, size))
    print(f"plotting the Q-function for action {action}")
    for xIdx, x in enumerate(np.linspace(5, -5, size)):
        print(f"{round(100*(xIdx/size))} % computed")
        for yIdx, y in enumerate(np.linspace(-5, 5, size)):
            state = tf.constant([[x, y, 0, 0]], dtype=tf.float32)
            Q = agent.criticMain(state, action)
            QArray[xIdx, yIdx] = Q
    plt.imshow(QArray, interpolation='nearest',
               cmap='hot', extent=[-5, 5, -5, 5])
    plt.colorbar()
    plt.title(f"Q-function for action {action}")
    plt.show()


def plotReward(episodeAvgScore) -> None:
    fig = plt.figure()
    plt.plot(episodeAvgScore)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
