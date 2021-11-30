import numpy as np
import random


"""
Code for prioritized experience replay mostly taken from:
https://medium.com/analytics-vidhya/building-a-powerful-dqn-in-tensorflow-2-0-explanation-tutorial-d48ea8f3177a
"""


class Memory(object):


    def __init__(self, size=1000000, input_shape=(105, 80), history_length=4, use_per=False):
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.use_per = use_per
        self.count = 0
        self.current = 0

        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal = np.empty(self.size, dtype=np.bool)
        if self.use_per:
            self.priorities = np.empty(self.size, dtype=np.float32)


    def add_experience(self, next_frame, action, reward, terminal):
        self.actions[self.current] = action
        self.frames[self.current, ...] = next_frame
        self.rewards[self.current] = reward
        self.terminal[self.current] = terminal
        if self.use_per:
            self.priorities[self.current] = max(self.priorities.max(), 1)

        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size, priority_scale=0.7):
        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        if self.use_per:
            scaled_priorities = self.priorities[self.history_length:self.count-1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                if self.use_per:
                    index = np.random.choice(np.arange(self.history_length, self.count-1), p=sample_probabilities)
                else:
                    index = random.randint(self.history_length, self.count - 1)

                if index >= self.current and index - self.history_length <= self.current:
                    continue
                if self.terminal[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx-self.history_length:idx, ...])
            new_states.append(self.frames[idx-self.history_length+1:idx+1, ...])

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1)).astype("float32") / 255.
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1)).astype("float32") / 255.

        if self.use_per:
            # Get importance weights from probabilities calculated earlier
            importance = 1/self.count * 1/sample_probabilities[[index - self.history_length for index in indices]]
            importance = importance / importance.max()

            return (states, self.rewards[indices], self.actions[indices], new_states, self.terminal[indices]),\
                    importance,\
                    indices

        else:
            return states, self.rewards[indices], self.actions[indices], new_states, self.terminal[indices]


    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


    def save(self, folder_name):
        """Save the replay buffer to a folder"""
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')