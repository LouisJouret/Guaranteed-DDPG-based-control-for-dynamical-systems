# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from networks.actor import Actor
from networks.critic import Critic


class Agent():
    def __init__(self) -> None:

        self.actorMain = Actor(stateDim=4, actionDim=2,
                               layer1Dim=128, layer2Dim=128)
        self.actorTarget = self.actorMain
        self.criticMain = Critic(self.actorMain)
        self.criticTarget = self.criticMain

        self.actorOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.criticOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.actorTarget.compile(optimizer=self.actorOptimizer)
        self.criticTarget.compile(optimizer=self.criticOptimizer)
        self.minAction = -1
        self.maxAction = 1
        self.gamma = 0.99
        self.tau = 0.005

        self.batchSize = 32
        self.buffer = []

    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actorMain(state)
        actions = tf.clip_by_value(actions, self.minAction, self.maxAction)
        return actions[0]

    def updateTarget(self):
        update = (1-self.tau) * self.actorTarget.get_weights() + \
            self.tau * self.actorMain.get_weights()
        self.actorTarget.set_weights(update)

    def train(self):
        states, statesNext, rewards, actions, dones = self.buffer.sample(
            self.batchSize)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        statesNext = tf.convert_to_tensor(statesNext, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # compute next action
            targetActions = self.actorTarget(statesNext)

            targetNextState = tf.squeeze(
                self.criticTarget(statesNext, targetActions), 1)
            qCritic = tf.squeeze(self.criticMain(states, actions), 1)
            qBellman = rewards + self.gamma * targetNextState * dones
            criticLoss = tf.keras.losses.MSE(qBellman, qCritic)

            newActions = self.actorMain(states)
            actorLoss = -self.criticMain(states, newActions)
            actorLoss = tf.math.reduce_mean(actorLoss)

        grads1 = tape1.gradient(
            actorLoss, self.actorMain.trainable_variables)
        grads2 = tape2.gradient(
            criticLoss, self.criticMain.trainable_variables)
        self.actorOptimizer.apply_gradients(
            zip(grads1, self.actorMain.trainable_variables))
        self.criticOptimizer.apply_gradients(
            zip(grads2, self.criticMain.trainable_variables))

        self.updateTarget()
