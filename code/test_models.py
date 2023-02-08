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


def test(agent: Agent):

    agent.actorMain.load_weights('./models/actor')

    system = doubleIntegrator(-2, -3, 4.5, 4)

    fig, ax = plt.subplots()
    ax.set_xlim(left=-5, right=5)
    ax.set_ylim(bottom=-5, top=5)
    ax.set_aspect('equal', adjustable='box')
    speed_up = 2
    buffer = []

    def animate(frame):
        action = agent.actorMain(system.state)
        x_new = system.step(system.state, action)
        system.state = x_new
        buffer.append(system.state)
        print(system.state)

    goal = plt.Circle((system.goal_x, system.goal_y),
                      system.successThreshold, color='blue')
    ax.add_patch(goal)

    anim = animation.FuncAnimation(
        fig, animate, frames=1000, repeat=False)
    plt.show()
    plt.close()
