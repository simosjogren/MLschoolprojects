# School work - Advanced Q-learning 1/2
# Simo Sjögren

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

def next_action(state):
    # Tells if we want to continue using the same routes or try to discover a new route.
    if np.random.uniform(0, 1) > epsilon:
        action = np.argmax(qtable[state,:])
    else:
        action = np.random.randint(0,4)
    return action

env = gym.make('Taxi-v3')

# Setting parameters
episodes = 1000 # num of training episodes
interactions = 100 # max num of interactions per episode
epsilon = 0.99 # e-greedy
alpha = 0.5 # learning rate - 1.
gamma = 0.9 # reward decay rate

# Initialize Q-table to zeros
num_states = env.observation_space.n
num_actions = env.action_space.n
qtable = np.zeros((num_states, num_actions))
print(qtable.shape)

# Set up performance tracking
episode_rewards = np.zeros(episodes)
window_size = 100
mean_rewards = np.zeros(episodes - window_size + 1)
std_rewards = np.zeros(episodes - window_size + 1)

# Main Q-learning loop
for episode in range(episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for interact in range(interactions):
        # exploitation vs. exploratin by e-greedy sampling of actions
        action = next_action(state)

        # Observe
        new_state, reward, done, info = env.step(action)

        # Update Q-table
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
                
        # Our new state is state
        state = new_state
        total_rewards += reward

        #  Reduce epsilon after each episode
        epsilon = epsilon * 0.99
            
        # Check if terminated
        if done == True: 
            break

    # Track performance
    episode_rewards[episode] = total_rewards
    if episode >= window_size:
        mean_rewards[episode - window_size] = np.mean(episode_rewards[episode - window_size:episode])
        std_rewards[episode - window_size] = np.std(episode_rewards[episode - window_size:episode])

# Plot performance
plt.figure()
plt.plot(mean_rewards)
plt.fill_between(range(episodes - window_size + 1), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
plt.title('Q-learning performance on Taxi-v3')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()

# For next exercise, we want to export the q-table into a dataset file:
with open('qtable.pkl', 'wb') as f:
    pickle.dump(qtable, f)