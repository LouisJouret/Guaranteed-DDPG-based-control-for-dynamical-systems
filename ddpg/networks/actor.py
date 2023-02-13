# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Lambda
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
        self.l1 = Dense(self.layer1Dim, activation='relu',
                        kernel_regularizer='l1_l2')
        self.l2 = Dense(self.layer2Dim, activation='relu',
                        kernel_regularizer='l1_l2')
        self.l3 = CustomLayer(self.actionDim)

    def __call__(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        return x


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs=2):
        super().__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])

    def call(self, x):
        x = tf.matmul(x, self.kernel)
        return pieceWiseLinear(x)
        # bool_up = tf.cast(x > 1, dtype=tf.bool)
        # bool_down = tf.cast(x < -1, dtype=tf.bool)
        # x = tf.where(bool_up, x, 0.001*x + 0.999)
        # x = tf.where(bool_down, x, 0.001*x - 0.999)
        # return x


@tf.custom_gradient
def pieceWiseLinear(x):
    bool_up = tf.cast(x > 1, dtype=tf.bool)
    bool_down = tf.cast(x < -1, dtype=tf.bool)
    x = tf.where(bool_up, x, 0.001*x + 0.999)
    x = tf.where(bool_down, x, 0.001*x - 0.999)

    def grad(dx):
        bool_up = tf.cast(dx > 1, dtype=tf.bool)
        bool_down = tf.cast(dx < -1, dtype=tf.bool)
        grad_dx = tf.ones(dx.shape)
        print(bool_up)
        grad_dx = tf.where(bool_up, grad_dx, 0.001)
        grad_dx = tf.where(bool_down, grad_dx, 0.001)
        return dx * grad_dx

    return x, grad
