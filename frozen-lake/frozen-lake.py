from copyreg import pickle

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle



def run(episodes, is_training=True, render = True):
    env = gym.make('FrozenLake-v1', render_mode="human" if render else None)
    rng = np.random.default_rng()

    states_count = env.observation_space.n
    actions_count = env.action_space.n

    if is_training:
        q = np.zeros((states_count, actions_count))
    else:
        f = open('frozen-lake.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    epsilon = 1
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon_decay_rate = 2 / episodes
    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)
    for i in range(episodes):
        print('Running episode ', i)
        state = env.reset()[0]
        rewards = 0
        steps = 0
        terminated = False
        while (not terminated and rewards > -1000):

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, _, _ = env.step(action)
            if is_training:
                q[state, action] = q[state, action] + learning_rate * (reward + discount_factor * np.max(q[new_state, :]) - q[state, action])

            rewards += reward
            state = new_state
            steps += 1

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = rewards
        steps_per_episode[i] = steps

    if is_training:
        f = open('frozen-lake.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

    env.close()
    if is_training:
        plot_rewards(rewards_per_episode, steps_per_episode)


def plot_rewards(rewards_per_episode, steps_per_episode):
    episodes = len(rewards_per_episode)
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):t + 1])

    fig, axs = plt.subplots(2)
    fig.suptitle('Reward and steps per training episode')
    axs[0].plot(mean_rewards, color = 'green')
    axs[1].plot(steps_per_episode, color = 'blue')
    plt.savefig(f'frozen-lake.png')

if __name__ == '__main__':
    #Train
    run(2000, is_training=True, render = False)

    #Test
    run(10, is_training=False, render=True)