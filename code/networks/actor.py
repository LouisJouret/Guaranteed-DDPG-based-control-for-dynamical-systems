# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Model


class Actor(Model):
    def __init__(self, stateDim, actionDim, layer1Dim=20, layer2Dim=20):
        super().__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.createModel()

    def createModel(self):
        "creates keras model of 2 dense layers followed by a sigmoid output"
        self.l1 = Dense(self.layer1Dim, activation='relu')
        self.l2 = Dense(self.layer2Dim, activation='relu')
        self.lAct = Dense(self.actionDim, activation='linear')

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.lAct(x)
        return x
