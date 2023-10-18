import gym
import time
import numpy as np
import matplotlib.pyplot as plt

episodes = 1000 # num of training episodes
interactions = 100 # max num of interactions per episode
epsilon = 0.99 # e-greedy Suuri epsilon = tutkii uusia mahdollisia reittejä vanhojen sijasta enemmän.
alpha = 0.5 # learning rate - 1. Suuri alpha = muuttaa Q-arvoja nopeasti.
gamma = 0.9 # reward decay rate. Suuri gamma = Pitkän aikavälin tähtäin.
debug = True # for non-slippery case to observe learning

env = gym.make("Taxi-v3")
env.reset()

actionspace = env.action_space.n
statespace = env.observation_space.n
qtable = np.zeros((statespace, actionspace))


def eval_policy_better(env_, pi_, gamma_, t_max_, episodes_):
    env_.reset()
    v_pi_rep = np.empty(episodes_)
    for e in range(episodes_):
        s_t = env.reset()
        v_pi = 0
        for t in range(t_max_):
            a_t = pi_[s_t]
            s_t, r_t, done, info = env_.step(a_t) 
            v_pi += gamma_**t*r_t
            if done:
                break
        v_pi_rep[e] = v_pi
        env.close()
    return np.mean(v_pi_rep), np.min(v_pi_rep), np.max(v_pi_rep), np.std(v_pi_rep)


def next_action(state):
    # Tells if we want to continue using the same routes or try to discover a new route.
    if np.random.uniform(0, 1) > epsilon:
        action = np.argmax(qtable[state,:])
    else:
        action = np.random.randint(0,4)
    return action


def main(epsilon):

    # Set up performance tracking
    episode_rewards = np.zeros(episodes)
    window_size = 100
    mean_rewards = np.zeros(episodes - window_size + 1)
    std_rewards = np.zeros(episodes - window_size + 1)
    # Q-learning loop
    hist = [] # evaluation history
    print("Starting Q-learning process.")
    for episode in range(episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        for interact in range(interactions):
            action = next_action(state)
            new_state, reward, done, info = env.step(action)
            # Update Q-table
            qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            state = new_state
            total_rewards += reward
            # env.render()
            epsilon = epsilon * 0.99

            if done == True:
                break
        # Track performance
        episode_rewards[episode] = total_rewards
        if episode >= window_size:
            mean_rewards[episode - window_size] = np.mean(episode_rewards[episode - window_size:episode])
            std_rewards[episode - window_size] = np.std(episode_rewards[episode - window_size:episode])
    print(qtable)
    # env.reset()

    hist = np.array(hist)
    print(hist.shape)

    '''plt.plot(hist[:,0],hist[:,1])
    plt.fill_between(hist[:,0], hist[:,1]-hist[:,4],hist[:,1]+hist[:,4],
        alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
        linewidth=0)
    plt.show()'''
    # Plot performance
    plt.figure()
    plt.plot(mean_rewards)
    plt.fill_between(range(episodes - window_size + 1), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    plt.title('Q-learning performance on Taxi-v3')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.show()

main(epsilon)