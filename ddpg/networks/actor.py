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
        "creates keras model of 2 dense layers followed by a custom piece-wise linear output"
        self.l1 = Dense(
            self.layer1Dim, activation=keras.layers.LeakyReLU(alpha=0.01))
        self.l2 = Dense(
            self.layer2Dim, activation=keras.layers.LeakyReLU(alpha=0.01))
        # self.l3 = PLULayer(self.actionDim)
        self.l3 = Dense(self.actionDim, activation='tanh')

    def __call__(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        return x


class PLULayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs=2):
        super().__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        intializer = tf.keras.initializers.GlorotNormal()
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs],
                                      regularizer=tf.keras.regularizers.l1_l2(), initializer=intializer)

    def call(self, x):
        x = tf.matmul(x, self.kernel)
        return pieceWiseLinear(x)


@ tf.custom_gradient
def pieceWiseLinear(x):
    bool_up_flag = tf.greater(x, 100)
    bool_down_flag = tf.less(x, -100)
    bool_up_flat = tf.cast(x > 1, dtype=tf.bool)
    bool_down_flat = tf.cast(x < -1, dtype=tf.bool)
    bool_up_semi = tf.cast(x > 0.5, dtype=tf.bool)
    bool_down_semi = tf.cast(x < -0.5, dtype=tf.bool)
    x = tf.where(bool_up_flag, x, 100)
    x = tf.where(bool_down_flag, x, -100)
    x = tf.where(bool_up_flat, x, 0.001*x + 0.999)
    x = tf.where(bool_down_flat, x, 0.001*x - 0.999)
    x = tf.where(bool_up_semi, x, 0.5*x + 0.25)
    x = tf.where(bool_down_semi, x, 0.5*x - 0.25)

    def grad(dx):
        stability = tf.cast(abs(x) > 100, dtype=tf.bool)
        bool_flat = tf.cast(abs(dx) > 1, dtype=tf.bool)
        bool_semi = tf.cast(abs(x) > 0.5, dtype=tf.bool)
        grad_dx = tf.ones(dx.shape)
        grad_dx = tf.where(stability, dx, 0)
        grad_dx = tf.where(bool_flat, grad_dx, 0.001)
        grad_dx = tf.where(bool_semi, grad_dx, 0.5)
        return dx * grad_dx

    return x, grad
