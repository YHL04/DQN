from utils import *
from dqn import Agent
from environment import Environment

from utils import save_frames_as_gif


env_name = "BreakoutDeterministic-v4"
env = Environment(env_name, training=False)

state_space = env.env.observation_space.shape
action_space = env.env.action_space.n

agent = Agent(state_size=state_space, action_size=action_space)
agent.model.summary()
agent.load_best(env_name)

testing_epochs = 10
best_reward = 0.
final_frames = []

for i in range(testing_epochs):
    frames = []

    done = False
    state = env.reset()
    state = agent.process_state(state)
    total_reward = 0.

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, life_lost = env.step(action)

        total_reward += reward
        reward = agent.process_reward(reward)
        next_state = agent.process_state(next_state)

        state = next_state

        scaled_frame = render(env, scale=3)
        frames.append(scaled_frame)

        if done:
            print(f"Testing Episode {i} \t Reward {total_reward}")
            if total_reward > best_reward:
                best_reward = total_reward
                final_frames = frames

save_frames_as_gif(final_frames)
