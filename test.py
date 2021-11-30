import gym
import time

from utils import *
from dqn import Agent

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env_name = "BreakoutDeterministic-v4"
env = gym.make(env_name)

state_space = env.observation_space.shape
action_space = env.action_space.n

agent = Agent(state_size=state_space, action_size=action_space)
agent.load(env_name, 15000)

print("Observation Space: ", state_space)
print("Action Space: ", action_space)
print("Action Meaning: ", env.unwrapped.get_action_meanings())


for i in range(10):
    done = False
    state = env.reset()
    state = agent.process_state(state)
    total_reward = 0

    first_step = True

    while not done:
        action = agent.get_action(state, training=False)
        if first_step:
            action = 1
            first_step = False
        next_state, reward, done, info = env.step(action)
        next_state = agent.process_state(next_state)
        reward = agent.process_reward(reward)
        total_reward += reward

        state = next_state

        render(env, scale=3)
        time.sleep(0.005)
        # for real time visualization

    print(f"Episode {i} \t Reward {total_reward}")