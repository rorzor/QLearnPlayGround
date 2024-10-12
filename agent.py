import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from environment import GridEnvironment
import numpy as np
import random
from collections import deque

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()
        self.replay_buffer = deque(maxlen=2000)  # Experience replay buffer


    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.choice(range(self.action_size))  # Explore
        q_values = self.model(state)
        return np.argmax(q_values[0]) 

    def train(self, state, action, reward, next_state, done, gamma):
        target = reward
        if not done:
            target = reward + gamma * np.amax(self.model(next_state)[0].numpy())

        target_f = self.model(state).numpy()
        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size, gamma):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + gamma * np.amax(self.model(next_state)[0].numpy())

            target_f = self.model(state).numpy()
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)