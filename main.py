import gym
from utils import *
from dqn import Agent
import time

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

env = gym.make("BreakoutDeterministic-v4")

state_space = env.observation_space.shape
action_space = env.action_space.n

agent = Agent(state_size=state_space, action_size=action_space)

print("Observation Space: ", state_space)
print("Action Space: ", action_space)

for i in range(50000):
    done = False
    state = env.reset()
    state = agent.process_state(state)
    total_reward = 0

    while not done:
        action = agent.get_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        next_state = agent.process_state(next_state)
        total_reward += reward

        agent.remember(state, action, reward, next_state, done)
        agent.train()
        next_state = state

        #if i % 20 == 0:
            # rendering slows down program dramatically
            # so only do it once in a while to demonstrate performance
        render(env, scale=3)

        if done:
            print(f"Episode {i} finished \t Reward {total_reward}")

    agent.update_target_model()
    if i % 200 == 0:
        agent.save("no", i)