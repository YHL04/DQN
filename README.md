# DQN

![alt text](https://github.com/YHL04/DQN/blob/master/test_gifs/results.gif "GIF")

Dueling Double DQN with e-greedy replaced with Noisy Nets:

![alt text](https://github.com/YHL04/DQN/blob/master/60hour_dueling_noisy_dqn.png "Plot")


Dueling Double DQN progress after 40+ hours before screen froze:

![alt text](https://github.com/YHL04/DQN/blob/master/40hour_dueling_dqn.png "Plot")


My guess for the instability of DQN is that since experience replay replaces the oldest memories with new ones, as the agent learns the optimal policy, the replay would mostly be popularized by the same states and overfitting to it. This results in a sudden collapse of the policy and then the agent quickly recovers while the replay is populated again by "bad" states that it has forgotten. (Someone explain to me if im wrong)
