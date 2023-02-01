# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from networks.actor import Actor
from networks.critic import Critic


class Agent():
    def __init__(self) -> None:
        self.actorMain = Actor()
        self.actorTarget = Actor()
        self.criticMain = Critic()
        self.criticTarget = Critic()
        self.batch_size = 64
        self.actionsDim = len(env.action_space.high)
        self.actorOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.criticOptimizer = tf.keras.optimizers.Adam(1e-4)

        self.actorTarget.compile(optimizer=self.actorOptimizer)
        self.criticTarget.compile(optimizer=self.criticOptimizer)
