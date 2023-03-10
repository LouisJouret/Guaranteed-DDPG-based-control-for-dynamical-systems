# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
import keras


class Critic(keras.Model):
    def __init__(self, stateDim, actionDim, layer1Dim=512, layer2Dim=512):
        super().__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.createModel()

    def createModel(self):
        "creates keras model of 2 dense layers followed by a sigmoid output"
        self.l1 = Dense(
            self.layer1Dim, activation=keras.layers.LeakyReLU(alpha=0.01))
        self.l2 = Dense(
            self.layer2Dim, activation=keras.layers.LeakyReLU(alpha=0.01))
        self.lq = Dense(1, activation=None)

    def call(self, state, action):
        x = self.l1(tf.concat([state, action], axis=1))
        x = self.l2(x)
        x = self.lq(x)
        return x

    def save_model(self):
        self.save_weights('ddpg/models/critic.h5')

    def load_model(self):
        self.load_weights('ddpg/models/critic.h5')
