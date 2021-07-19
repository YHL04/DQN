import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import os
from collections import deque

import time


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 1e-4

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.buffer_size = 2000
        self.state_buffer = np.zeros(
            (self.buffer_size, self.state_size[0], self.state_size[1], self.state_size[2]*4), dtype="float32")
        self.next_state_buffer = np.zeros(
            (self.buffer_size, self.state_size[0], self.state_size[1], self.state_size[2]*4), dtype="float32")
        self.reward_buffer = np.zeros((self.buffer_size,), dtype="int32")
        self.action_buffer = np.zeros((self.buffer_size,), dtype="int32")
        self.done_buffer = np.zeros((self.buffer_size,), dtype="bool")

        self.pointer = 0
        self.cap = 0

        self.state = deque(maxlen=4)
        for i in range(4):
            self.state.append(np.zeros(self.state_size))


    def get_action(self, state, training):
        if training and random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        output = self.model(state)
        return np.argmax(output)


    def process_state(self, state):
        self.state.append(state)
        return np.concatenate(self.state, axis=2)


    def build_model(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.state_size[0], self.state_size[1], self.state_size[2]*4)),
                keras.layers.Conv2D(32, kernel_size=(8, 8), strides=4, activation="relu"),
                keras.layers.Conv2D(32, kernel_size=(4, 4), strides=2, activation="relu"),
                keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation="relu"),
                keras.layers.Flatten(),

                keras.layers.Dense(1024, activation="relu"),
                keras.layers.Dense(self.action_size, activation="linear")
            ]
        )
        model.compile(loss="mse", optimizer="adam")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.state_buffer[self.pointer] = state
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.next_state_buffer[self.pointer] = next_state
        self.done_buffer[self.pointer] = done

        self.pointer += 1
        if self.pointer >= self.buffer_size:
            self.pointer = 0

        self.cap += 1
        if self.cap >= self.buffer_size:
            self.cap = self.buffer_size


    def train(self):
        batch_indices = np.random.choice(self.cap, self.batch_size)

        state_batch = self.state_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]
        done_batch = self.done_buffer[batch_indices]

        self.update(state_batch, reward_batch, action_batch, next_state_batch, done_batch)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def update(self, state, reward, action, next_state, done):
        start=time.time()
        q_value = self.target_model.predict(next_state)
        print('q val ', time.time() - start)

        y = reward + self.gamma * np.max(q_value, axis=1)
        y = np.where(done, reward, y)
        print('numpy ', time.time() - start)

        target = self.model.predict(state)
        target[:, action] = y
        print('state ', time.time()-start)

        start = time.time()
        self.model.train_on_batch(state, target)
        print('train', time.time() - start)


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name, i):
        path = "saves/{}{}/".format(name, i)
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save_weights(path + "model.h5")
        self.target_model.save_weights(path + "target_model.h5")

