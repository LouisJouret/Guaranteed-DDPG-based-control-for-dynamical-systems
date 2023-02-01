# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from tensorflow.python.keras.layers import Dense, Flatten, Lambda
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import Model
import numpy as np


class actor(Model):
    def __init__(self, inputDim, actionDim, upper_bounds, layer1Dim=512, layer2Dim=512):
        super().__init__()
        self.inputDim = inputDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.actionDim = actionDim
        self.upper_bounds = upper_bounds

    def createModel(self):
        "creates keras model of 2 dense layers followed by a sigmoid output"
        model = Sequential()
        model.add(Dense(self.layer1Dim, input_dim=self.inputDim, activation='relu'))
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
