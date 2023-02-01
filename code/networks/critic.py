# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from tensorflow.python.keras.layers import Dense, Flatten, Lambda
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import Model
import numpy as np


class Critic(Model):
    "constructor of the critic network"

    def __init__(self, inputDim, actionDim, upper_bounds, layer1Dim=512, layer2Dim=512):
        super().__init__()
        self.inputDim = inputDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.actionDim = actionDim
        self.upper_bounds = upper_bounds
        self.model = self.createModel()
        self.buffer = []
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.005
        self.update_target_model = self.tau * self.update_target_model
        self.update_model = self.tau * self.update_model
        self.loss = 'mse'
        self.optimizer = 'adam'

    def createModel(self):
        "creates keras model for the critic network"
        model = Sequential()
        model.add(Dense(self.layer1Dim, input_dim=self.inputDim, activation='relu'))
        model.add(Dense(self.layer2Dim, activation='relu'))
        # no activation function for the critic
        model.add(Dense(self.actionDim, activation=None))

    def train(self):
        pass


class CriticTarget():
    def __init__(self, critic) -> None:
        self.inputDim = critic.inputDim
        self.layer1Dim = critic.layer1Dim
        self.layer2Dim = critic.layer2Dim
        self.actionDim = critic.actionDim
        self.upper_bounds = critic.upper_bounds
        self.targetmodel = self.createModel()
        self.targetModel.set_weights(critic.get_weights())
        self.buffer = []
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.005
        self.loss = 'mse'
        self.optimizer = 'adam'

    def createModel(self):
        "creates keras model for the critic network"
        model = Sequential()
        model.add(Dense(self.layer1Dim, input_dim=self.inputDim, activation='relu'))
        model.add(Dense(self.layer2Dim, activation='relu'))
        # no activation function for the critic
        model.add(Dense(self.actionDim, activation=None))

    def train(self, modelMain):
        "updates target model with weights of main model"

        mainWeights = modelMain.get_weights()
        targetWeights = self.targetModel.get_weights()
        targetWeights = [self.tau * mainWeight + (1 - self.tau) *
                         targetWeight for mainWeight, targetWeight in
                         zip(mainWeights, targetWeights)]
        self.targetModel.set_weights(targetWeights)
