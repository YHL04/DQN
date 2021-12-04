import matplotlib.pyplot as plt
import time

from utils import *
from dqn import Agent
from environment import Environment


env_name = "BreakoutDeterministic-v4"
log = open(f'logs/{env_name}.txt', 'w')

env = Environment(env_name, training=True)

state_space = env.env.observation_space.shape
action_space = env.env.action_space.n

agent = Agent(state_size=state_space, action_size=action_space)

print("Observation Space: ", state_space)
print("Action Space: ", action_space)
print("Action Meaning: ", env.env.unwrapped.get_action_meanings())

agent.model.summary()
ep_reward = []

training_epochs = 500000
frames_elapsed = 0
start = time.time()

for i in range(training_epochs):
    done = False
    state = env.reset()
    state = agent.process_state(state)
    total_reward = 0.
    total_loss = 0.

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, life_lost = env.step(action)


        total_reward += reward
        reward = agent.process_reward(reward)
        next_state = agent.process_state(next_state)

        agent.remember(next_state, action, reward, life_lost)
        loss = agent.train()
        total_loss += loss

        state = next_state

        render(env, scale=3)
        frames_elapsed += 1

        if frames_elapsed % 10000 == 0:
            agent.update_target_model()

        if done:
            print(f"Episode {i} finished \t Reward {total_reward} \t Loss {total_loss} \t Frames {frames_elapsed} \t Time Elapsed {time.time()-start}")
            log.write(f"{i}, {total_reward}, {total_loss}, {frames_elapsed}\n")
            log.flush()
            ep_reward.append(total_reward)

    if i % 5000 == 0:
        agent.save(env_name, i)

log.close()