# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from networks.actor import Actor
from networks.critic import Critic
from buffer import RBuffer
from noise import OhrsteinUhlenbeckNoise
from keras.optimizers import Adam
import numpy as np
import keras


class Agent():
    def __init__(self, num_actions, num_states) -> None:
        self.actionDim = num_actions
        self.stateDim = (num_states,)
        self.actorMain = Actor(self.stateDim, self.actionDim,
                               layer1Dim=512, layer2Dim=512)
        self.actorTarget = Actor(self.stateDim, self.actionDim,
                                 layer1Dim=512, layer2Dim=512)
        self.criticMain = Critic(self.stateDim, 1,
                                 layer1Dim=512, layer2Dim=512)
        self.criticTarget = Critic(self.stateDim, 1,
                                   layer1Dim=512, layer2Dim=512)

        self.actorOptimizer = Adam(learning_rate=1e-3)
        self.criticOptimizer = Adam(learning_rate=2e-3)

        self.gamma = 0.99
        self.tau = 0.005

        self.batchSize = 64
        self.maxBufferSize = 1000000

        self.ounoise = OhrsteinUhlenbeckNoise(
            np.zeros(self.actionDim), np.array([0.1] * self.actionDim))

        self.replayBuffer = RBuffer(maxsize=self.maxBufferSize,
                                    statedim=self.actorMain.stateDim,
                                    naction=self.actorMain.actionDim)

        self.actorTarget.compile(optimizer=self.actorOptimizer)
        self.criticTarget.compile(optimizer=self.criticOptimizer)
        self.actorMain.compile(optimizer=self.actorOptimizer)
        self.criticMain.compile(optimizer=self.criticOptimizer)

        self.updateActorTarget(1)
        self.updateCriticTarget(1)

    def act(self, state):
        actions = self.actorMain(state)
        actions += self.ounoise()
        # actions += tf.random.normal(shape=[self.actionDim],
        #                             mean=0.0, stddev=0.05)
        actions = tf.clip_by_value(actions, -1, 1)
        return actions

    def updateActorTarget(self, tau):
        weights = []
        target_weights = self.actorTarget.trainable_variables
        for i, weight in enumerate(self.actorMain.trainable_variables):
            weights.append(tau * weight + (1-tau)
                           * target_weights[i])
            target_weights[i].assign(weights[i])
        # tf.numpy_function(self.target_actor.set_weights, tf.Variable(weights), tf.float32)

    def updateCriticTarget(self, tau):
        weights = []
        target_weights = self.criticTarget.trainable_variables
        for i, weight in enumerate(self.criticMain.trainable_variables):
            weights.append(tau * weight + (1-tau)
                           * target_weights[i])
            target_weights[i].assign(weights[i])
        # tf.numpy_function(self.target_critic.set_weights, tf.Varibale(weights), tf.float32)

    def train(self):
        if self.replayBuffer.cnt < self.batchSize:
            return

        states, actions, rewards, dones, nextStates = self.replayBuffer.sample(
            self.batchSize)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        nextStates = tf.convert_to_tensor(nextStates, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        actions = tf.reshape(actions, (self.batchSize, self.actionDim))

        self.backprop(states, actions, rewards, dones, nextStates)

    @tf.function
    def backprop(self, states, actions, rewards, dones, nextStates):

        with tf.GradientTape() as tape1:
            actionNext = self.actorTarget(nextStates)
            qNext = tf.squeeze(self.criticTarget(nextStates, actionNext))
            qCritic = tf.squeeze(self.criticMain(states, actions), 1)
            qBellman = rewards + self.gamma * qNext * (1-dones)
            criticLoss = keras.losses.MSE(qCritic, qBellman)

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

        self.updateActorTarget(self.tau)
        self.updateCriticTarget(self.tau)

    def save(self):
        print('Saving models...')
        self.actorMain.save_weights('code/models/actor.h5')
        self.actorTarget.save_weights('code/models/actor_target.h5')
        self.criticMain.save_weights('code/models/critic.h5')
        self.criticTarget.save_weights('code/models/critic_target.h5')
