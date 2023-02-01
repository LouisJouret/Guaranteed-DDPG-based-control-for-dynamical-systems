# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Lambda, Input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import Model
import numpy as np
import keras.backend as kerasBackend


class Actor(Model):
    def __init__(self, stateDim, actionDim, batchSize, layer1Dim=512, layer2Dim=512):
        super().__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.batchSize = batchSize
        self.numHiddenLayers = 2
        self.learningRate = 0.001

        self.model, self.modelWeights, self._model_input = \
            self.createModel()
        self.targetModel, self.targetWeights, self._target_state = \
            self.createModel()
        self.targetModel.set_weights(self.modelWeights)

    def createModel(self):
        "creates keras model of 2 dense layers followed by a sigmoid output"
        inputLayer = Input(shape=[self.stateDim])
        model = Sequential()
        model.add(Dense(self.layer1Dim, input_dim=self.stateDim, activation='relu'))
        model.add(Dense(self.layer2Dim, activation='relu'))
        model.add(Dense(self.actionDim, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def trainTargetModel(self, model, targetModel, buffer):
        "updates target model with weights of model"
        for i in range(len(buffer)):
            state, action, reward, next_state, done = buffer[i]
            targetModel.fit(np.array([state]), np.array(
                [action]), epochs=1, verbose=0)
            targetModel.fit(np.array([next_state]), np.array(
                [reward]), epochs=1, verbose=0)
        targetModel.set_weights(model.get_weights())

    def trainModel(self, model, buffer):
        "updates model with buffer"
        pass
