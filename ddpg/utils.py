# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from __future__ import annotations
import matplotlib.pyplot as plt
from agent import Agent
import tensorflow as tf
import polytope as pc
import numpy as np
import random
import keras


def plotQ(agent: Agent, iter) -> None:
    "plots a 2D canvas of the Q-function for a given state and action"
    size = 50
    QArray = np.zeros((size, size))
    print(f"plotting the Q-function for best action ...")
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
    plt.savefig(f"ddpg/figures/episode_{iter}_Q.png")
    plt.close()


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
                         cmap='hot', extent=[-5, 5, -5, 5])
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"u_x(x,y)")
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(AYArray, interpolation='nearest',
                         cmap='hot', extent=[-5, 5, -5, 5])
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    plt.xlabel("x")
    plt.title("u_y(x,y)")
    plt.savefig(f"ddpg/figures/episode_{iter}_value.png")
    plt.close()


def plotActionVectors(agent: Agent, env, iter) -> None:
    "plots a 2D canvas of the x input for a given state"
    print("Generating action vectors ...")
    size = 20
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    XArray = np.zeros((size, size))
    YArray = np.zeros((size, size))
    AXArray = np.zeros((size, size))
    AYArray = np.zeros((size, size))
    for xIdx, x in enumerate(np.linspace(-5, 5, size)):
        for yIdx, y in enumerate(np.linspace(5, -5, size)):
            state = tf.constant([[x, y]], dtype=tf.float32)
            action = agent.act(state)
            AXArray[yIdx, xIdx] = action[0][0]
            AYArray[yIdx, xIdx] = action[0][1]
            XArray[yIdx, xIdx] = x
            YArray[yIdx, xIdx] = y
    fig = plt.figure()
    plt.quiver(XArray, YArray, AXArray, AYArray)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('square')
    plt.title(f"action vectors for episode {iter}")
    plt.savefig(f"ddpg/figures/episode_{iter}_vectors.png")
    plt.close()


def plotLinearRegion(agent: Agent, iter) -> None:
    "plots the linear regions of the Neural Network"

    print("Generating the linear regions of the action space ...")
    A_border = np.array([[1.0, 0.0],
                        [-1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, -1.0]])
    b_border = 5 * np.array([1.0, 1.0, 1.0, 1.0])
    border_polytope = pc.Polytope(A_border, b_border)
    box = pc.bounding_box(border_polytope)

    regions = [RegionReLU(border_polytope, old_S=np.identity(2),
                          old_w_actif=np.identity(2), old_b_actif=np.zeros((2, 1)))]

    for layer in [agent.actorMain.l1, agent.actorMain.l2, agent.actorMain.l3]:
        weights = layer.weights
        new_regions = []
        for region in regions:
            region.compute_activation_weights(weights)
            region.cut()
            region.compute_S()
            region.inherite_actif_para()
            for kid in region.kids:
                new_regions.append(kid)
        regions = new_regions

    regionsPlu = []
    for region in regions:
        regionPlu = RegionPLU(polytope=region.polytope, old_S=region.old_S,
                              old_w_actif=region.old_w_actif, old_b_actif=region.old_b_actif)
        regionsPlu.append(regionPlu)
    weights = agent.actorMain.lact.weights
    new_regions = []
    for region in regionsPlu:
        region.compute_activation_weights(weights)
        region.cut()
        region.compute_S()
        region.inherite_actif_para()
        for kid in region.kids:
            new_regions.append(kid)
    regions = new_regions

    fig, ax = plt.subplots()
    for region in new_regions:
        region.polytope.plot(ax, linewidth=0.5, linestyle='--')
    ax.set_xlim(box[0][0]-1, box[1][0] + 1)
    ax.set_ylim(box[0][1]-1, box[1][1] + 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('square')
    plt.title(f"{len(regions)} linear regions for episode {iter}")
    plt.savefig(f"ddpg/figures/episode_{iter}_linear_regions.png")
    plt.close()


class RegionReLU:
    def __init__(self, polytope: pc.Polytope, old_S=None, old_w_actif=None, old_b_actif=None):
        self.polytope = polytope
        self.old_w_actif = old_w_actif
        self.old_b_actif = old_b_actif
        self.w_actif = None
        self.b_actif = None
        self.kids = []  # list of sub regions
        self.old_S = old_S
        self.cuts = []

    def cut(self):
        W = self.w_actif
        B = self.b_actif
        for neuron in range(W.shape[0]):
            w = np.array(W[neuron, :])
            b = np.array(B[neuron])
            cut = pc.Polytope(np.vstack((self.polytope.A, -w)),
                              np.append(self.polytope.b, b))
            self.cuts.append(cut)

        self.kids = [RegionReLU(self.polytope)]
        for cut in self.cuts:
            copy = list(self.kids)
            for sub_region in copy:
                first_kid = RegionReLU(sub_region.polytope.intersect(cut))
                if first_kid.polytope.volume > 0:
                    second_kid = RegionReLU(
                        sub_region.polytope.diff(first_kid.polytope))
                    if second_kid.polytope.volume > 0:
                        self.kids.append(first_kid)
                        self.kids.append(second_kid)
                        self.kids.remove(sub_region)

    def compute_activation_weights(self, weights):
        W = np.transpose(weights[0])
        B = np.transpose([weights[1]])
        self.w_actif = np.matmul(W, np.matmul(self.old_S, self.old_w_actif))
        self.b_actif = np.matmul(W, np.dot(
            self.old_S, self.old_b_actif)) + B

    def compute_S(self):
        for kid in self.kids:
            S = []
            for cut in self.cuts:
                if kid.polytope == pc.intersect(kid.polytope, cut):
                    S.append(1)
                else:
                    S.append(0)
            kid.old_S = np.diag(S)

    def inherite_actif_para(self):
        for kid in self.kids:
            kid.old_w_actif = self.w_actif
            kid.old_b_actif = self.b_actif


class RegionPLU(RegionReLU):

    def __init__(self, polytope: pc.Polytope, old_S=None, old_w_actif=None, old_b_actif=None):
        super().__init__(polytope, old_S, old_w_actif, old_b_actif)

    def cut(self):
        W = self.w_actif
        B = self.b_actif
        for neuron in range(W.shape[0]):
            w = np.array(W[neuron, :])
            b = np.array(B[neuron])
            cut1 = pc.Polytope(np.vstack((self.polytope.A, -w)),
                               np.append(self.polytope.b, b - 1))
            cut2 = pc.Polytope(np.vstack((self.polytope.A, w)),
                               np.append(self.polytope.b, b - 0.5))
            cut3 = pc.Polytope(np.vstack((self.polytope.A, -w)),
                               np.append(self.polytope.b, b + 0.5))
            cut4 = pc.Polytope(np.vstack((self.polytope.A, w)),
                               np.append(self.polytope.b, b + 1))
            self.cuts.append(cut1)
            self.cuts.append(cut2)
            self.cuts.append(cut3)
            self.cuts.append(cut4)

        self.kids = [RegionPLU(self.polytope)]
        for cut in self.cuts:
            copy = list(self.kids)
            for sub_region in copy:
                first_kid = RegionPLU(sub_region.polytope.intersect(cut))
                if first_kid.polytope.volume > 0:
                    second_kid = RegionPLU(
                        sub_region.polytope.diff(first_kid.polytope))
                    if second_kid.polytope.volume > 0:
                        self.kids.append(first_kid)
                        self.kids.append(second_kid)
                        self.kids.remove(sub_region)


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
