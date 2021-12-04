import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import random
import os
from collections import deque

from model import build_model
from memory import Memory

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class Agent:
    def __init__(self, state_size, action_size,
                 use_per=False,
                 batch_size=32,
                 learning_rate=0.00005,
                 gamma=0.99,
                 n_stack=4):


        self.state_size = state_size
        self.action_size = action_size
        self.use_per = use_per

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.n_stack = n_stack
        self.memory = Memory(use_per=use_per)
        self.state = deque(maxlen=n_stack)

        self.model = build_model(state_size, action_size, n_stack, learning_rate)
        self.target_model = build_model(state_size, action_size, n_stack, learning_rate)
        self.update_target_model()

        self.pointer = 0
        self.cap = 0


    def get_action(self, state):
        state = np.expand_dims(state, axis=-1)
        self.state.append(state)

        if len(self.state) != 4:
            for i in range(4):
                self.state.append(state)

        state = np.concatenate(self.state, axis=2)
        state = np.expand_dims(state, axis=0)
        state = state.astype("float32")
        state /= 255.
        output = self.model.predict(state)
        return np.argmax(output)


    def process_state(self, state):
        state = self.to_grayscale(self.downsample(state))
        return state

    def process_reward(self, reward):
        #reward = np.sign(reward)
        return reward

    def to_grayscale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)

    def downsample(self, img):
        return img[::2, ::2]

    def remember(self, next_state, action, reward, done):
        self.memory.add_experience(next_state, action, reward, done)

    def train(self, priority_scale=1.0):
        if self.memory.count <= self.memory.history_length:
            return 0.

        if self.use_per:
            (states, rewards, actions, new_states, done), importance, indices \
                = self.memory.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)
            importance = importance ** (1 - self.eps)
            loss = self.update(states, rewards, actions, new_states, done, importance, indices)
        else:
            states, rewards, actions, new_states, done \
                = self.memory.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)
            loss = self.update(states, rewards, actions, new_states, done)

        return loss

    def update(self, state, reward, action, next_state, done, importance=None, indices=None):
        target = self.model.predict(state)
        q_value = np.max(self.target_model.predict(next_state), axis=1)

        y = reward + self.gamma * q_value
        y = np.where(done, reward, y)
        target[np.arange(self.batch_size), action] = y

        loss = self.model.train_on_batch(state, target)
        return loss


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name, i):
        path = f"saves/{name}/episode_{i}/"
        self.model.load_weights(path + "model.h5")

    def load_best(self, name):
        max_i = 0
        assert len(os.listdir(f"saves/{name}")) != 0
        for file in os.listdir(f"saves/{name}"):
            i = int(file.split("_")[-1])
            if i > max_i:
                max_i = i

        print(max_i)

        path = f"saves/{name}/episode_{max_i}/"
        self.model.load_weights(path + "model.h5")


    def save(self, name, i):
        file = f"saves/{name}/"
        path = file + f"episode_{i}/"
        if not os.path.exists(file):
            os.mkdir(file)
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save_weights(path + "model.h5")
