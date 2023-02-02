# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from networks.actor import Actor
from networks.critic import Critic
from buffer import RBuffer
import random
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np


class Agent():
    def __init__(self) -> None:

        self.actorMain = Actor(stateDim=4, actionDim=2,
                               layer1Dim=128, layer2Dim=128)
        self.actorTarget = self.actorMain
        self.criticMain = Critic(self.actorMain)
        self.criticTarget = self.criticMain
        self.actorOptimizer = Adam(learning_rate=1e-4)
        self.criticOptimizer = Adam(learning_rate=1e-4)
        self.actorTarget.compile(optimizer=self.actorOptimizer)
        self.criticTarget.compile(optimizer=self.criticOptimizer)
        self.minAction = -1
        self.maxAction = 1
        self.gamma = 0.99
        self.tau = 0.005

        self.batchSize = 10
        self.maxBufferSize = 1000

        self.replayBuffer = RBuffer(maxsize=self.maxBufferSize,
                                    statedim=self.actorMain.stateDim,
                                    naction=self.actorMain.actionDim)

    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actorMain(state)
        actions = tf.clip_by_value(actions, self.minAction, self.maxAction)
        return actions[0]

    def updateActorTarget(self):
        weights = []
        targets = self.actorTarget.weights
        for i, weight in enumerate(self.actorMain.weights):
            weights.append(weight * (1 - self.tau) + targets[i]*self.tau)
        self.actorTarget.set_weights(weights)

    def updateCriticTarget(self):
        weights = []
        targets = self.criticTarget.weights
        for i, weight in enumerate(self.criticMain.weights):
            weights.append(weight * self.tau + targets[i]*(1-self.tau))
        self.criticTarget.set_weights(weights)

    def train(self):
        if self.replayBuffer.cnt < self.batchSize:
            return

        states, actions, rewards, dones, nextStates = self.replayBuffer.sample(
            self.batchSize)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

            nextAction = self.actorTarget(nextStates)
            qNext = self.criticTarget(nextStates, nextAction)

            qCritic = self.criticMain(states, actions)
            qBellman = rewards + self.gamma * qNext
            criticLoss = tf.keras.losses.MSE(qBellman, qCritic)

            newActions = self.actorMain(nextStates)
            actorLoss = -self.criticMain(nextStates, newActions)  # minus!
            actorLoss = tf.math.reduce_mean(actorLoss)

        grads1 = tape1.gradient(
            actorLoss, self.actorMain.trainable_variables)
        grads2 = tape2.gradient(
            criticLoss, self.criticMain.trainable_variables)
        self.actorOptimizer.apply_gradients(
            zip(grads1, self.actorMain.trainable_variables))
        self.criticOptimizer.apply_gradients(
            zip(grads2, self.criticMain.trainable_variables))

        self.updateActorTarget()
        self.updateCriticTarget()
