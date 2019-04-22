# Mountain car using DQN
# https://github.com/openai/gym/wiki/MountainCar-v0
# https://github.com/rlcode/reinforcement-learning/blob/master/4-gym/1-mountaincar/mountaincar_dqn.py

import gym
import numpy as np
import pylab
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODE = 1000
ENV_NAME = 'MountainCar-v0'

class CarAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.005
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 50000
        self.learning_rate = 0.001
        self.train_start = 1000
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        if action == 2:
            action = 1
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # print("act_values {}".format(act_values))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, batch_size)
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state)[0]
            if not done:
                target[action] = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            else:
                target[action] = reward
            update_input[i] = state
            update_target[i] = target
            # fit the model
        self.model.fit(update_input, update_target
                       ,batch_size=batch_size, epochs=1, verbose=0)
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    action_size = 2
    print(action_size)
    agent = CarAgent(state_size, action_size)
    agent.load_model("./save/MoutainCar_DQN.h5")
    done = False
    batch_size = 64
    train_start = 1000
    scores, episodes = [], []


    for i_episode in range(EPISODE):
        done = False
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # print(state)
        score = 0
        fake_action = 0
        action_count = 0

        # for t in range(200):
        while not done:
            # env.render()
            action_count += 1
            if action_count == 4:
                action = agent.act(state)
                action_count = 0
                if action == 0:
                    fake_action = 0
                elif action == 1:
                    fake_action = 2
            # print("action {}".format(action))
            next_state, reward, done, info = env.step(fake_action)
            next_state = np.reshape(next_state, [1, state_size])

            # reward = reward if not done else -100
            agent.remember(state, fake_action, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
            score += reward
            if done:
                env.reset()
                agent.update_target_model()
                scores.append(score)
                episodes.append(i_episode)
                # print(scores, episodes)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./MountainCar_DQN.png")
                print("Episode {}, score {}, memory {}, epsilon {}"
                      .format(i_episode, score, len(agent.memory), agent.epsilon))

        if i_episode % 50 == 0:
            agent.save_model("./save/MoutainCar_DQN.h5")
    env.close()






