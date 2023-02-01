# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Model
from actor import Actor


class Critic(Model):
    def __init__(self, actor: Actor):
        super().__init__()
        self.stateDim = actor.stateDim
        self.actionDim = actor.actionDim
        self.layer1Dim = actor.layer1Dim
        self.layer2Dim = actor.layer2Dim
        self.batchSize = actor.batchSize
        self.learningRate = 0.001
        self.createModel()

    def createModel(self):
        "creates keras model of 2 dense layers followed by a sigmoid output"
        self.l1 = Dense(self.layer1Dim, activation='relu')
        self.l2 = Dense(self.layer2Dim, activation='relu')
        self.lAct = Dense(self.actionDim, activation=None)

    def call(self, state, action):
        x = self.l1(tf.concat([state, action], axis=1))
        x = self.l2(x)
        x = self.lAct(x)
        return x
