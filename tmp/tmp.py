from copyreg import pickle
from math import trunc

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

MAX_STEPS_PER_EPISODE = 200
PASSENGER_LOCATIONS_COUNT = 4

def run(episodes, is_training=True, render=True):
    env = gym.make('Taxi-v3', render_mode="human" if render else None)
    rng = np.random.default_rng()

    actions_count = env.action_space.n
    subtasks_count = 2  #0. Go to passenger, 1. Go to destination
    maze_len = 5
    pax_locations = PASSENGER_LOCATIONS_COUNT

    q_table = get_q_table(is_training, actions_count, maze_len, pax_locations)

    epsilon = 1
    learning_rate = 1
    discount_factor = 1
    epsilon_decay_rate = 1 / episodes
    rewards_per_episode = np.zeros((PASSENGER_LOCATIONS_COUNT, episodes))
    steps_per_episode = np.zeros(episodes)

    for i in range(episodes):

        (state, info) = env.reset()

        rewards = 0
        steps = 0
        terminated = False
        current_task = 0
        task_completed = False
        tx, ty, passenger_location, destination = get_position_dim(state)

        print(f'\nRunning episode {i}, with passenger location {passenger_location},  destination {destination} and taxi coordinates ({tx}, {ty})')

        target = passenger_location
        while (steps < MAX_STEPS_PER_EPISODE and not terminated and not task_completed):
            steps += 1
            extra_reward = 0
            tx, ty, _, _ = get_position_dim(state)

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample(info["action_mask"])
            else:
                if current_task == 0:
                    action = np.argmax(q_table[tx, ty, passenger_location, :])
                if current_task == 1:
                    action = np.argmax(q_table[tx, ty, destination, :])

                # action = np.argmax(q[tx, ty, passenger_location, np.where(info["action_mask"] == 1)[0]])

            if is_taxi_at_destination(tx, ty, passenger_location) and current_task == 0:
                extra_reward = 21
                print(f'Got to passenger in {steps} steps, coordinates ({tx}, {ty})')
                current_task = 1  # switch to new task
                if action != 4:
                    print(f'Choose {action} instead of 4')
                    action = 4

            if is_taxi_at_destination(tx, ty, destination) and current_task == 1:
                extra_reward = 21
                print(f'Got to destination in {steps} steps, coordinates ({tx}, {ty})')
                task_completed = True
                target = destination
                if action != 5:
                    print(f'Choose {action} instead of 5')
                    action = 5 #Drop the passenger

            new_state, reward, terminated, _, info = env.step(action)

            tx_new, ty_new, _, _ = get_position_dim(new_state)

            reward += extra_reward

            if is_training:
                q_table[tx, ty, target, action] = q_table[tx, ty, target, action] + learning_rate * \
                                                        (reward + discount_factor * np.max(
                                                            q_table[tx_new, ty_new, target, :]) - q_table[
                                                             tx, ty, target, action])
                rewards += reward

            state = new_state

        print(
            f'Completed episode {i} in {steps} steps with reward {rewards} and completion {terminated} and epsilon {epsilon}')
        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, 0)
            rewards_per_episode[target][i] = rewards
            steps_per_episode[i] = steps

    if is_training:
        f = open('taxi.pkl', 'wb')
        pickle.dump(q_table, f)
        f.close()

        plot_rewards(rewards_per_episode)

    env.close()



def is_taxi_at_destination(tx, ty, destination):
    return tx * 5 + ty == get_pass_position_index_by_location(destination)


def get_q_table(is_training, actions_count, maze_len, pax_locations):
    if is_training:
        q = np.zeros((maze_len, maze_len, pax_locations, actions_count))
    else:
        f = open('taxi.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    return q


def get_pass_position_index_by_location(location):
    if location == 0:
        return 0
    if location == 1:
        return 4
    if location == 2:
        return 20
    if location == 3:
        return 23
    else:
        return -1


def get_position_dim(position):
    destination = position % 4
    poz = position // 4
    passenger_location = poz % 5
    taxi_location = poz // 5
    taxi_col = taxi_location % 5
    taxi_row = taxi_location // 5
    return (taxi_row, taxi_col, passenger_location, destination)


def plot_rewards(rewards_per_episode):
    episodes = len(rewards_per_episode[0])
    targets = PASSENGER_LOCATIONS_COUNT
    mean_rewards = np.zeros((targets, episodes))
    for i in range(targets):
        for t in range(episodes):
            mean_rewards[i][t] = np.mean(rewards_per_episode[i][max(0, t - 100):t + 1])

    fig, axs = plt.subplots(targets + 1)
    fig.suptitle('Reward per training episode')
    colors = ['red', 'green', 'blue', 'orange', 'black']
    for i in range(targets):
        axs[i].plot(mean_rewards[i], color=colors[i])

    plt.savefig(f'taxi.png')


if __name__ == '__main__':
    # Train
    #run(5001, is_training=True, render = False)

    # Test
    run(10, is_training=False, render=True)