import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import os
from collections import deque

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = 16
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999999

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.buffer_size = 30000
        self.state_buffer = np.zeros(
            (self.buffer_size, self.state_size[0]//2, self.state_size[1]//2, 4), dtype="uint32")
        self.next_state_buffer = np.zeros(
            (self.buffer_size, self.state_size[0]//2, self.state_size[1]//2, 4), dtype="uint32")
        # self.state_buffer = np.zeros((self.buffer_size, self.state_size[0]), dtype="float64")
        # self.next_state_buffer = np.zeros((self.buffer_size, self.state_size[0]), dtype="float64")
        self.reward_buffer = np.zeros((self.buffer_size,), dtype="int32")
        self.action_buffer = np.zeros((self.buffer_size,), dtype="int32")
        self.done_buffer = np.zeros((self.buffer_size,), dtype="bool")

        self.pointer = 0
        self.cap = 0

        self.state = deque(maxlen=4)
        for i in range(4):
            self.state.append(np.zeros((self.state_size[0]//2, self.state_size[1]//2, 1)))


    def get_action(self, state, training):
        if training and random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        state = state.astype("float32")
        state /= 255.
        output = self.model.predict(state)
        return np.argmax(output)


    def process_state(self, state):
        #return state
        state = self.to_grayscale(self.downsample(state))
        state = np.expand_dims(state, axis=-1)
        self.state.append(state)
        return np.concatenate(self.state, axis=2)

    def process_reward(self, reward):
        #reward = np.sign(reward)
        return reward

    def to_grayscale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)

    def downsample(self, img):
        return img[::2, ::2]


    def build_model(self):
        # model = keras.Sequential(
        #     [
        #         keras.layers.Input(shape=self.state_size),
        #         keras.layers.Dense(64, activation="relu"),
        #         keras.layers.Dense(64, activation="relu"),
        #         keras.layers.Dense(64, activation="relu"),
        #         keras.layers.Dense(self.action_size, activation="linear")
        #     ]
        # )
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.state_size[0]//2, self.state_size[1]//2, 4)),
                keras.layers.Conv2D(32, kernel_size=(8, 8), strides=4, padding="same", activation="relu"),
                keras.layers.Conv2D(32, kernel_size=(4, 4), strides=2, padding="same", activation="relu"),
                keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding="same", activation="relu"),
                keras.layers.Flatten(),

                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(self.action_size, activation="linear")
            ]
        )
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=1e-4))
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

        state_batch = self.state_buffer[batch_indices].astype("float32") / 255.
        reward_batch = self.reward_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices].astype("float32") / 255.
        done_batch = self.done_buffer[batch_indices]

        loss = self.update(state_batch, reward_batch, action_batch, next_state_batch, done_batch)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss


    def update(self, state, reward, action, next_state, done):
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
        self.target_model.load_weights(path + "target_model.h5")

    def save(self, name, i):
        file = f"saves/{name}/"
        path = file + f"episode_{i}/"
        if not os.path.exists(file):
            os.mkdir(file)
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save_weights(path + "model.h5")
        self.target_model.save_weights(path + "target_model.h5")
