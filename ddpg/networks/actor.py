# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, InputLayer
import keras


class Actor(keras.Model):
    def __init__(self, stateDim, actionDim, layer1Dim=512, layer2Dim=512, layer3Dim=512):
        super().__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.layer3Dim = layer3Dim
        self.createModel()

    def createModel(self):
        "creates keras model of 3 dense layers followed by a custom piece-wise linear output"
        self.l0 = InputLayer(input_shape=self.stateDim)
        self.l1 = Dense(
            self.layer1Dim, activation=keras.layers.LeakyReLU(alpha=0.001))
        self.l2 = Dense(
            self.layer2Dim, activation=keras.layers.LeakyReLU(alpha=0.001))
        self.lact = PLULayer(self.actionDim)

    def call(self, state):
        x = self.l0(state)
        x = self.l1(x)
        x = self.l2(x)
        x = self.lact(x)
        return x

    def save_model(self):
        self.save_weights('ddpg/models/actor.h5')

    def load_model(self):
        self.load_weights('ddpg/models/actor.h5')


class PLULayer(tf.keras.layers.Layer):
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


@ tf.custom_gradient
def pieceWiseLinear(x):
    # bool_up_flag = tf.cast(x > 100, dtype=tf.bool)
    # bool_down_flag = tf.cast(x < -100, dtype=tf.bool)
    # bool_up_flat = tf.cast(x > 1.5, dtype=tf.bool)
    # bool_down_flat = tf.cast(x < -1.5, dtype=tf.bool)
    # bool_up_semi = tf.cast(x > 0.5, dtype=tf.bool)
    # bool_down_semi = tf.cast(x < -0.5, dtype=tf.bool)
    # x = tf.where(bool_up_semi, x, 0.5*x + 0.25)
    # x = tf.where(bool_down_semi, x, 0.5*x - 0.25)
    # x = tf.where(bool_up_flat, x, 0.001*x + 0.999)
    # x = tf.where(bool_down_flat, x, 0.001*x - 0.999)
    # x = tf.where(bool_up_flag, x, 1.099)
    # x = tf.where(bool_down_flag, x, -1.099)
    # print(x)
    bool_up_flag = tf.math.greater(tf.abs(x), 100*tf.ones_like(x))
    bool_up_flat = tf.math.greater(tf.abs(x), 1.5*tf.ones_like(x))
    bool_up_semi = tf.math.greater(tf.abs(x), 0.5*tf.ones_like(x))
    x = tf.where(bool_up_semi, 0.5*x + tf.sign(x)*0.25, x)
    x = tf.where(bool_up_flat, 0.001*x + tf.sign(x)*0.999, x)
    x = tf.where(bool_up_flag, tf.sign(x)*1.099, x)

    def grad(upstream):
        stability = tf.math.greater(tf.abs(x), 100*tf.ones_like(x))
        bool_flat = tf.math.greater(tf.abs(x), 1.5*tf.ones_like(x))
        bool_semi = tf.math.greater(tf.abs(x), 0.5*tf.ones_like(x))
        dy_dx = tf.ones(x.shape)
        dy_dx = tf.where(bool_semi, 0.5, dy_dx)
        dy_dx = tf.where(bool_flat, 0.001, dy_dx)
        dy_dx = tf.where(stability, 0.0, dy_dx)

        return dy_dx * upstream

    return x, grad
