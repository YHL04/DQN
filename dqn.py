import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add, Subtract, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

import numpy as np
import random
import os
from collections import deque

from memory import Memory

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class Agent:
    def __init__(self, state_size, action_size,
                 use_per=False,
                 batch_size=32,
                 learning_rate=0.00005,
                 gamma=0.99,
                 eps_initial=1,
                 eps_mid=0.1,
                 eps_final=0.01,
                 eps_decay_interval_1=1000000,
                 eps_decay_interval_2=2000000,
                 n_stack=4):


        self.state_size = state_size
        self.action_size = action_size
        self.use_per = use_per

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        # linear epsilon decay to eps_mid over interval 1, then to eps_final over interval 2
        self.eps_initial = eps_initial
        self.eps = eps_initial
        self.eps_mid = eps_mid
        self.eps_final = eps_final
        self.eps_decay_interval_1 = eps_decay_interval_1
        self.eps_decay_interval_2 = eps_decay_interval_2

        self.n_stack = n_stack
        self.memory = Memory(use_per=use_per)
        self.state = deque(maxlen=n_stack)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.pointer = 0
        self.cap = 0


    def decay_epsilon(self):
        if self.eps > self.eps_mid:
            self.eps -= (self.eps_initial-self.eps_mid) / self.eps_decay_interval_1
        elif self.eps > self.eps_final:
            self.eps -= (self.eps_mid-self.eps_final) / self.eps_decay_interval_2
        else:
            self.eps = self.eps_final

    def get_action(self, state, training):
        state = np.expand_dims(state, axis=-1)
        self.state.append(state)
        if len(self.state) != 4:
            for i in range(4):
                self.state.append(state)
        state = np.concatenate(self.state, axis=2)

        if training and random.uniform(0, 1) <= self.eps:
            return random.randrange(self.action_size)

        state = np.expand_dims(state, axis=0)
        state = state.astype("float32")
        state /= 255.
        output = self.model.predict(state)
        return np.argmax(output)

    def get_random_action(self, state):
        state = np.expand_dims(state, axis=-1)
        self.state.append(state)
        return random.randrange(self.action_size)


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

    def build_model(self):
        input  = Input(shape=(self.state_size[0]//2, self.state_size[1]//2, self.n_stack))
        x      = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', padding="same", use_bias=False)(input)
        x      = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
        x      = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
        x      = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

        # Split into value and advantage streams
        val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)

        val_stream = Flatten()(val_stream)
        val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

        adv_stream = Flatten()(adv_stream)
        adv = Dense(self.action_size, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

        # Combine streams into Q-Values
        reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

        output = Add()([val, Subtract()([adv, reduce_mean(adv)])])

        # Build model
        model = Model(input, output)
        model.compile(loss=Huber(), optimizer=Adam(lr=self.learning_rate))

        return model


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

        self.decay_epsilon()

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

    def save(self, name, i):
        file = f"saves/{name}/"
        path = file + f"episode_{i}/"
        if not os.path.exists(file):
            os.mkdir(file)
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save_weights(path + "model.h5")
