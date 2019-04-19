# Mountain car using DQN

import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODE = 1000

class CarAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, nex_state, done):
        self.memory.append((state, action, reward, nex_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randn(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])



if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = CarAgent(state_size, action_size)
    done = False
    batch_size = 32

    print(action_size, state_size)
    for i_episode in range(EPISODE):
        observation = env.reset()
        print(observation)
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            obslist = list((observation, action, reward, done, info))
            print(obslist)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
        print("End episode {}".format(i_episode))

    env.close()






