# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
import keras


class Actor(keras.Model):
    def __init__(self, stateDim, actionDim, layer1Dim=512, layer2Dim=512):
        super().__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.createModel()

    def createModel(self):
        "creates keras model of 2 dense layers followed by a sigmoid output"
        # initializer = tf.keras.initializers.GlorotNormal(seed=165835)
        self.l1 = Dense(
            self.layer1Dim, activation='relu')
        self.l2 = Dense(
            self.layer2Dim, activation='relu')
        self.lAct = Dense(self.actionDim, activation='tanh')

    def __call__(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.lAct(x)
        return x
