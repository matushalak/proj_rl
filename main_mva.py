from collections import deque

from env import DataCenterEnv
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path

environment = DataCenterEnv(path_to_dataset)

aggregate_reward = 0
terminated = False
state = environment.observation()

window_size = 23 #a month
price_buffer = deque(maxlen=window_size)


while not terminated:
    # agent is your own imported agent class
    # action = agent.act(state)
    #action = np.random.uniform(-1, 1)
    #action = -1
    # next_state is given as: [storage_level, price, hour, day]


    #add price to price_buffer for mva
    current_price = state[1]
    price_buffer.append(current_price)
    #update mva
    if len(price_buffer) < window_size:
        moving_average = sum(price_buffer) / len(price_buffer)
    else:
        moving_average = sum(price_buffer) / window_size

    """ """
    if current_price < moving_average:
        action = np.random.uniform(0, 1)
        action=1
    elif current_price > moving_average:
        action = np.random.uniform(-1, 0)
        action = -1
    else:
        action = 0

    #action = np.random.uniform(-1, 1)
    next_state, reward, terminated = environment.step(action)

    state = next_state
    aggregate_reward += reward
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)

print('Total reward:', aggregate_reward)