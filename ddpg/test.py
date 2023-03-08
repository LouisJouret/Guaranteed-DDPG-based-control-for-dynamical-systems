#  Copyright(c) 2023 Louis Jouret
#
#  This software is released under the MIT License.
#  https: // opensource.org/licenses/MIT

import numpy as np
from agent import Agent
from gymEnv import Mouse

env = Mouse()
agent = Agent(len(env.actions), len(env.observations))
# run the model with a dummy input to initialize the weights
dummy_state = np.zeros((1, len(env.observations)))
agent.actorMain(dummy_state)

episodes = 100

agent.actorMain.load_model()
print(agent.actorMain.summary())

for episode in range(episodes):
    done = False
    observation = env.reset()
    while not done:
        env.render()
        action = agent.actorMain(np.array([observation]))
        nextObservation, reward, done, _ = env.step(action[0])
        observation = nextObservation
