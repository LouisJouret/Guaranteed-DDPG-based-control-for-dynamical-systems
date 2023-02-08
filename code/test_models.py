# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from agent import Agent
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from dynamicModel import doubleIntegrator
import keras
import json
from keras.models import model_from_json


def animate(agent: Agent):

    agent.actorMain.load_weights('./models/actor')

    system = doubleIntegrator(-2, -3, 4.5, 4)

    fig, ax = plt.subplots()
    ax.set_xlim(left=-5, right=5)
    ax.set_ylim(bottom=-5, top=5)
    ax.set_aspect('equal', adjustable='box')
    speed_up = 2
    buffer = []
    dot = plt.Circle((-2, -3), 0.75, color='black')

    def animate(frame):
        done = False
        while not done:
            action = agent.actorMain(system.state)
            x_new, reward, done = system.step(system.state, action)
            buffer.append(system.state)
            dot.center = (x_new[0], x_new[1])
            buffer.append(x_new)

    goal = plt.Circle((system.goal_x, system.goal_y),
                      system.successThreshold, color='blue')
    ax.add_patch(goal)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(buffer), repeat=False)
    plt.show()
    plt.close()
