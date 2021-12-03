import gym


class Environment:
    '''
    DeepMind sets terminal as true if life is lost to boost DQN performance
    '''
    def __init__(self, env_name, auto_start=True):
        self.env = gym.make(env_name)
        self.last_lives = 0

        self.auto_start = auto_start
        self.fire = False

    def reset(self):
        if self.auto_start:
            self.fire = True
        return self.env.reset()

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

