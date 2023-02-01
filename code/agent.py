# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from networks.actor import Actor
from networks.critic import Critic
import random
from tf_agents.replay_buffers import tf_uniform_replay_buffer


class Agent():
    def __init__(self) -> None:

        self.actorMain = Actor(stateDim=4, actionDim=2,
                               layer1Dim=128, layer2Dim=128)
        self.actorTarget = self.actorMain
        self.criticMain = Critic(self.actorMain)
        self.criticTarget = self.criticMain

        self.actorOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.criticOptimizer = tf.keras.optimizers.Adam(1e-4)
        # self.actorTarget.compile(optimizer=self.actorOptimizer)
        # self.criticTarget.compile(optimizer=self.criticOptimizer)
        self.minAction = -1
        self.maxAction = 1
        self.gamma = 0.99
        self.tau = 0.005

        self.batchSize = 32
        self.maxBufferSize = 1000

        dataSpec = (
            tf.TensorSpec([1, 4], tf.float32, 'state'),
            tf.TensorSpec([1, 2], tf.float32, 'action'),
            tf.TensorSpec([1], tf.float32, 'reward'),
            tf.TensorSpec([1], tf.float32, 'done'),
            tf.TensorSpec([1, 4], tf.float32, 'nextState'),
        )
        self.replayBuffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            dataSpec,
            batch_size=self.batchSize,
            max_length=self.maxBufferSize)

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
        if len(self.buffer) < self.batchSize:
            return
        memory = random.sample(self.buffer, self.batchSize)

        states = tf.convert_to_tensor(
            [datapoint[:4] for datapoint in memory])
        actions = tf.convert_to_tensor(
            [datapoint[4:6] for datapoint in memory])
        rewards = tf.convert_to_tensor(
            [datapoint[6] for datapoint in memory])
        statesNext = tf.convert_to_tensor(
            [datapoint[7:-1] for datapoint in memory])
        dones = tf.convert_to_tensor(
            [datapoint[-1] for datapoint in memory])

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

    def remember(self, state, action, reward, stateNext, done):
        done = [done]
        reward = [reward]
        list = [state, action, reward, stateNext, done]
        flat_list = [item for sublist in list for item in sublist]
        self.buffer.append(flat_list)
