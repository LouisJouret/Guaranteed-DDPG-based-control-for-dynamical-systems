# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Concatenate
from tensorflow.python.keras import Model
from networks.actor import Actor
import math


class Critic(Model):
    def __init__(self, actor: Actor):
        super().__init__()
        self.stateDim = actor.stateDim
        self.actionDim = actor.actionDim
        self.layer1Dim = actor.layer1Dim
        self.layer2Dim = actor.layer2Dim
        self.learningRate = 0.001
        self.createModel()

    def createModel(self):
        "creates keras model of 2 dense layers followed by a sigmoid output"
        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=math.sqrt(2/(self.actionDim + self.stateDim + 1))/10)
        self.lconcat = Concatenate()
        self.l1 = Dense(self.layer1Dim, activation='relu',
                        kernel_initializer=initializer, kernel_regularizer='l1_l2')
        self.l2 = Dense(self.layer2Dim, activation='relu',
                        kernel_initializer=initializer, kernel_regularizer='l1_l2')
        self.lq = Dense(1, activation=None, kernel_initializer=initializer,
                        kernel_regularizer='l1_l2')

    def call(self, state, action):
        x = self.lconcat([state, action])
        x = self.l1(x)
        x = self.l2(x)
        x = self.lq(x)
        return x
