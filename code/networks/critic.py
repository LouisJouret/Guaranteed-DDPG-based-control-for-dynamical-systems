# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from tensorflow.python.keras.layers import Dense, Flatten, Lambda
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import Model
import numpy as np


class critic():
    "constructor of the critic network"

    def __init__(self, inputDim, actionDim, upper_bounds, layer1Dim=512, layer2Dim=512):
        self.inputDim = inputDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.actionDim = actionDim
        self.upper_bounds = upper_bounds
        self.model = self.createModel()
        self.targetModel = self.createModel()
        self.targetModel.set_weights(self.model.get_weights())
        self.buffer = []
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.005
        self.update_target_model = self.tau * self.update_target_model
        self.update_model = self.tau * self.update_model
        self.loss = 'mse'
        self.optimizer = 'adam'

    def createModel(self):
        "creates keras model of 2 dense layers followed by a sigmoid output"
        model = Sequential()
        model.add(Dense(self.layer1Dim, input_dim=self.inputDim, activation='relu'))
        model.add(Dense(self.layer2Dim, activation='relu'))
        model.add(Dense(self.actionDim, activation='sigmoid'))
        model.compile(loss=self.loss, optimizer=self.optimizer)
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
