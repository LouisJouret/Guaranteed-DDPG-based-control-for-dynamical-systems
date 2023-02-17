# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
from agent import Agent
import tensorflow as tf
import numpy as np
import random


def plotQ(agent: Agent) -> None:
    "plots a 2D canvas of the Q-function for a given state and action"
    size = 50
    QArray = np.zeros((size, size))
    print(f"plotting the Q-function for best action")
    for xIdx, x in enumerate(np.linspace(-6, 6, size)):
        print(f"{round(100*(xIdx/size))} % computed")
        for yIdx, y in enumerate(np.linspace(6, -6, size)):
            state = tf.constant([[x, y, 0, 0]], dtype=tf.float32)
            action = agent.act(state)
            Q = agent.criticMain(state, action)
            QArray[yIdx, xIdx] = Q
    plt.imshow(QArray, interpolation='nearest',
               cmap='hot', extent=[-6, 6, -6, 6])
    plt.colorbar()
    plt.title(f"Q-function for best action")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.show()


def plotAction(agent: Agent, iter) -> None:
    "plots a 2D canvas of the x input for a given state"
    print("Generating action space figure ...")
    size = 50
    AXArray = np.zeros((size, size))
    AYArray = np.zeros((size, size))
    for xIdx, x in enumerate(np.linspace(-6, 6, size)):
        # print(f"{round(100*(xIdx/size))} % computed")
        for yIdx, y in enumerate(np.linspace(6, -6, size)):
            state = tf.constant([[x, y]], dtype=tf.float32)
            action = agent.act(state)
            AXArray[yIdx, xIdx] = action[0][0]
            AYArray[yIdx, xIdx] = action[0][1]
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(AXArray, interpolation='nearest',
                         cmap='hot', extent=[-6, 6, -6, 6])
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"u_x(x,y)")
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(AYArray, interpolation='nearest',
                         cmap='hot', extent=[-6, 6, -6, 6])
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    plt.xlabel("x")
    # plt.ylabel("y")
    plt.title("u_y(x,y)")
    plt.savefig(f"ddpg/figures/episode_{iter}.png")
    plt.close()


def plotReward(episodeAvgScore) -> None:
    fig = plt.figure()
    plt.plot(episodeAvgScore)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()


def getInitialPoint() -> None:
    x0 = 0
    y0 = 0
    while np.sqrt(x0**2 + y0**2) < 1:
        x0 = random.randint(-500, 500)/100
        y0 = random.randint(-500, 500)/100
    return x0, y0
