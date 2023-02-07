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
        self.actionDim = 1
        self.stateDim = 2
        self.actorMain = Actor(self.stateDim, self.actionDim,
                               layer1Dim=32, layer2Dim=32)
        self.actorTarget = Actor(self.stateDim, self.actionDim,
                                 layer1Dim=32, layer2Dim=32)
        self.criticMain = Critic(self.stateDim, 1,
                                 layer1Dim=32, layer2Dim=32)
        self.criticTarget = Critic(self.stateDim, 1,
                                   layer1Dim=32, layer2Dim=32)

        self.actorOptimizer = Adam(learning_rate=1e-4)
        self.criticOptimizer = Adam(learning_rate=1e-3)

        self.gamma = 0.99
        self.tau = 0.05

        self.batchSize = 64
        self.maxBufferSize = 1000

        self.replayBuffer = RBuffer(maxsize=self.maxBufferSize,
                                    statedim=self.actorMain.stateDim,
                                    naction=self.actorMain.actionDim)

        self.actorTarget.compile(optimizer=self.actorOptimizer)
        self.criticTarget.compile(optimizer=self.criticOptimizer)
        self.actorMain.compile(optimizer=self.actorOptimizer)
        self.criticMain.compile(optimizer=self.criticOptimizer)

        self.actorTarget.set_weights(self.actorMain.get_weights())
        self.criticTarget.set_weights(self.criticMain.get_weights())

    def act(self, state):
        # state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actorMain(state)
        # actions += tf.random.normal(shape=[self.actionDim],
        #                             mean=0.0, stddev=0.001, dtype=tf.float32)
        actions = tf.round(actions)
        return tf.clip_by_value(actions + 1, 0, 2)

    def updateActorTarget(self):
        weights = []
        target_weights = self.actorTarget.trainable_variables
        for i, weight in enumerate(self.actorMain.trainable_variables):
            weights.append(self.tau * weight + (1-self.tau)
                           * target_weights[i])
            target_weights[i].assign(weights[i])
        # tf.numpy_function(self.target_actor.set_weights, tf.Variable(weights), tf.float32)

    def updateCriticTarget(self):
        weights = []
        target_weights = self.criticTarget.trainable_variables
        for i, weight in enumerate(self.criticMain.trainable_variables):
            weights.append(self.tau * weight + (1-self.tau)
                           * target_weights[i])
            target_weights[i].assign(weights[i])
        # tf.numpy_function(self.target_critic.set_weights, tf.Varibale(weights), tf.float32)

    def train(self):
        if self.replayBuffer.cnt < self.batchSize:
            return

        states, actions, rewards, _, nextStates = self.replayBuffer.sample(
            self.batchSize)

        # states = tf.convert_to_tensor(states, dtype=tf.float32)
        # nextStates = tf.convert_to_tensor(nextStates, dtype=tf.float32)
        # rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        # actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        actions = tf.reshape(actions, (self.batchSize, 1))
        with tf.GradientTape() as tape1:
            actionNext = self.act(nextStates)
            actionNext = tf.reshape(actionNext, (self.batchSize, 1))
            qNext = self.criticTarget(nextStates, np.array(actionNext))
            qNext = tf.squeeze(qNext)
            qCritic = self.criticMain(states, actions)
            qBellman = rewards + self.gamma * qNext

            criticLoss = tf.keras.losses.MSE(qCritic, qBellman)

        with tf.GradientTape() as tape2:
            newAction = self.actorMain(states)
            Q = self.criticMain(states, newAction)
            actorLoss = tf.reduce_mean(-Q)

        gradsCritic = tape1.gradient(
            criticLoss, self.criticMain.trainable_variables)
        self.criticOptimizer.apply_gradients(
            zip(gradsCritic, self.criticMain.trainable_variables))

        gradsActor = tape2.gradient(
            actorLoss, self.actorMain.trainable_variables)
        self.actorOptimizer.apply_gradients(
            zip(gradsActor, self.actorMain.trainable_variables))

        self.updateActorTarget()
        self.updateCriticTarget()
