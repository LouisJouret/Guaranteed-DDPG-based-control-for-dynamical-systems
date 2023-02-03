# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Lambda
from tensorflow.python.keras import Model
import math


class Actor(Model):
    def __init__(self, stateDim, actionDim, layer1Dim=10, layer2Dim=10):
        super().__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.createModel()
        self.upperBound = 1

    def createModel(self):
        "creates keras model of 2 dense layers followed by a sigmoid output"
        # initializer = tf.keras.initializers.GlorotNormal()
        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=math.sqrt(2/(self.actionDim + self.stateDim))/10)
        self.l1 = Dense(self.layer1Dim, activation='relu',
                        use_bias=True, kernel_initializer=initializer, kernel_regularizer='l1_l2')
        self.l2 = Dense(self.layer2Dim, activation='relu',
                        use_bias=True, kernel_initializer=initializer, kernel_regularizer='l1_l2')
        self.lAct = Dense(self.actionDim, activation=None,
                          use_bias=True, kernel_initializer=initializer, kernel_regularizer='l1_l2')
        self.bound_layer = Lambda(lambda x: tf.clip_by_value(
            x, -self.upperBound, self.upperBound))

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.lAct(x)
        x = self.bound_layer(x)
        return x
