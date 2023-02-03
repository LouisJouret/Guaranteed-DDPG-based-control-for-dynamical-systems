# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
from agent import Agent
import tensorflow as tf
import numpy as np
import random


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
    print(QArray)
    plt.imshow(QArray, interpolation='nearest',
               cmap='hot', extent=[-5, 5, -5, 5])
    plt.colorbar()
    plt.title(f"Q-function for action {action}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plotReward(episodeAvgScore) -> None:
    fig = plt.figure()
    plt.plot(episodeAvgScore)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()


def getInitialPoint() -> None:
    x0 = 0
    y0 = 0
    while abs(x0) < 0.2 and abs(y0) < 0.2:
        x0 = random.randint(-500, 500)/100
        y0 = random.randint(-500, 500)/100
    return x0, y0
