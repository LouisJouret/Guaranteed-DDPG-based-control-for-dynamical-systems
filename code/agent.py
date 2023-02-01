# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from networks.actor import Actor
from networks.critic import Critic


class Agent():
    def __init__(self) -> None:

        self.actorMain = Actor(stateDim=4, actionDim=2, batchSize=1)
        self.actorTarget = self.actorMain
        self.criticMain = Critic(self.actorMain)
        self.criticTarget = self.criticMain

        self.actorOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.criticOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.actorTarget.compile(optimizer=self.actorOptimizer)
        self.criticTarget.compile(optimizer=self.criticOptimizer)
        self.minAction = -1
        self.maxAction = 1

        self.buffer = []

    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actorMain(state)
        actions = tf.clip_by_value(actions, self.minAction, self.maxAction)
        return actions[0]

    def updateTarget(self, tau):
        update = (1-tau) * self.actorTarget.get_weights() + \
            tau * self.actorMain.get_weights()
        self.actorTarget.set_weights(update)

    def train(self):
        states, next_states, rewards, actions, dones = self.buffer.sample(
            self.batch_size)
