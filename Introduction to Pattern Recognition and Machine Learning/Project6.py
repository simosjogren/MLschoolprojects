# School project - Q-learning practice on the FrozenLake-v1 practice
#

import gym
import random
import numpy as np
import matplotlib.pyplot as plt

SLIPPERY = False

env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY)
env.reset()
env.render()


def eval_policy(qtable_, num_of_episodes_, max_steps_):
    rewards = []
    for episode in range(num_of_episodes_):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        for step in range(max_steps_):
            action = np.argmax(qtable_[state,:])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward
            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    env.close()
    avg_reward = sum(rewards)/num_of_episodes_
    return avg_reward


def Q_learning(episodes, max_steps, reward_best=-1000):
    # Tune-up parameters
    alfa = 0.5
    gamma = 0.9

    action_size = env.action_space.n
    state_size = env.observation_space.n
    rewards = []
    rewards_by_episodes = []
    qtable = np.zeros((state_size, action_size))

    for episode in range(episodes):
        state = env.reset()
        done = False

        for step in range(max_steps):
            if np.max(qtable[state]) > 0:
                action = np.argmax(qtable[state])
            else:
                action = env.action_space.sample()
            
            new_state, reward, done, info = env.step(action)
            qtable[state, action] = qtable[state, action] + \
                                    alfa * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
            state = new_state

        if episode % 100 == 0:
            best_reward_after = eval_policy(qtable,1000,100)
            print(f'Best reward after episode {episode+1} is {best_reward_after}')
            rewards.append(best_reward_after)
            rewards_by_episodes.append(episode)
    
    return rewards, rewards_by_episodes


def plot_list(to_plot):
    print("Now plotting...")
    for index in range(len(to_plot)):

        episodes = np.array(to_plot[index][1])
        rewards = np.array(to_plot[index][0])
        plt.plot(episodes, rewards)
    plt.xlabel("Amount of episodes")
    plt.ylabel("Accuracy")
    plt.show()


def main():
    TOTAL_EPISODES = 1000
    MAX_STEPS = 200
    to_plot = []
    for n in range(10):
        print()
        print("Run number:", str(n+1))
        to_plot.append(Q_learning(TOTAL_EPISODES, MAX_STEPS))
    plot_list(to_plot)
    print("Test finished.")


if __name__ == "__main__":
    main()


def Q_learning_slippery(episodes, max_steps, reward_best=-1000):
    def update_ql_stochastic(q_table, state, action, reward, new_state, learn_r, gamma):
        q_table[state, action] = q_table[state, action] + learn_r * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        return q_table

    # for N_episodes in 
    # next_step = random.randint(0,3)
    action_size = env.action_space.n
    state_size = env.observation_space.n
    rewards = []
    rewards_by_episodes = []
    for episode in range(episodes):
        state = env.reset()
        step = 0
        done = False
        qtable = np.random.rand(state_size, action_size)
        # 16 saraektta , 4 riviä, floatti 0-1
        reward_tot = 0
        for step in range(max_steps):
            action = np.argmax(qtable[state,:])
            new_state, reward, done, info = env.step(action)

            qtable = update_ql_stochastic(qtable, state, action, reward, new_state, 0.5, 0.9)

            reward_tot += reward
            state = new_state
            # if done:
            #     break
        if reward_tot > reward_best:
            reward_best = reward_tot
            qtable_best = qtable
            print(f'Better found - reward: {reward_best}')
        if episode % 100 == 0:
            best_reward_after = eval_policy(qtable_best,1000,100)
            print(f'Best reward after episode {episode+1} is {best_reward_after}')
            rewards.append(best_reward_after)
            rewards_by_episodes.append(episode)
    print(f'Tot reward of the found policy: {eval_policy(qtable_best,1000,100)}')
    plot_list(rewards, rewards_by_episodes)
    return qtable_best

