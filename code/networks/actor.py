# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Lambda
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import Model


class Actor(Model):

    def __init__(self, n_states, n_actions, upper_bounds, n_layer1=512, n_layer2=512):
        super().__init__()
        self.n_states = n_states
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.n_actions = n_actions
        self.upper_bounds = upper_bounds

        self.layer1 = Dense(n_layer1, activation="relu")
        self.layer2 = Dense(n_layer2, activation="relu")
        self.layer3 = Dense(n_actions, activation="tanh")
        self.bound_layer = Lambda(lambda x: x * upper_bounds)

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        action = self.bound_layer(self.layer3(x))

        return action
