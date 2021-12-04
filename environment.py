import gym
import random


class Environment:
    '''
    DeepMind sets terminal as true if life is lost to boost DQN performance
    '''
    def __init__(self, env_name, auto_start=True, training=True, no_op_max=50):
        self.env = gym.make(env_name)
        self.last_lives = 0

        self.auto_start = auto_start
        self.fire = False

        self.training = training
        self.no_op_max = no_op_max

    def reset(self):
        if self.auto_start:
            self.fire = True
        frame = self.env.reset()

        if not self.training:
            for i in range(random.randint(1, self.no_op_max)):
                frame, _, _, _ = self.env.step(1)

        return frame

    def step(self, action):
        if self.fire:
            action = 1
            self.fire = False
        frame, reward, terminal, info = self.env.step(action)

        if info['ale.lives'] < self.last_lives:
            life_lost = True
            self.fire = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']

        return frame, reward, terminal, life_lost

