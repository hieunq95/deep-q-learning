# Mountain car using DQN

import gym
from collections import deque
from keras.models import Sequential

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

if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    env.reset()
    for i_episode in range(EPISODE):
        observation = env.reset()
        print(observation)
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            obslist = list((observation, reward, done, info))
            print(obslist)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
        print("End episode {}".format(i_episode))

    env.close()






