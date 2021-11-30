import gym


class Environment:
    '''
    DeepMind sets terminal as true if life is lost to boost DQN performance
    '''
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.last_lives = 0

    def reset(self):
        return self.env.reset()

    def step(self, action):
        frame, reward, terminal, info = self.env.step(action)

        if info['ale.lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']

        return frame, reward, terminal, life_lost

