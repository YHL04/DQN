import gym
import matplotlib.pyplot as plt

from utils import *
from dqn import Agent

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env_name = "BreakoutDeterministic-v4"
log = open(f'logs/{env_name}.txt', 'w')

env = gym.make(env_name)

state_space = env.observation_space.shape
action_space = env.action_space.n

agent = Agent(state_size=state_space, action_size=action_space)
agent.load("")
agent.update_target_model()

print("Observation Space: ", state_space)
print("Action Space: ", action_space)

checkpoint = agent.model.get_weights()
high_score = 0
ep_reward = []


for i in range(50000):
    done = False
    state = env.reset()
    state = agent.process_state(state)
    total_reward = 0
    total_loss = 0

    while not done:
        action = agent.get_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        next_state = agent.process_state(next_state)
        reward = agent.process_reward(reward)
        total_reward += reward

        agent.remember(state, action, reward, next_state, done)
        loss = agent.train()
        total_loss += loss
        state = next_state

        render(env, scale=3)
