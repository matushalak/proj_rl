from env import DataCenterEnv
import numpy as np
import argparse
from utils import preprocess_state
from agents import HourAgent, WeekdayAgent

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path

environment = DataCenterEnv(path_to_dataset)
# dates
timestamps = environment.timestamps

aggregate_reward = 0
terminated = False
state = environment.observation()
# adds relevant features so now each state is: 
# [storage_level, price, hour, day, calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month]
state = preprocess_state(state, timestamps)
print("Starting state:", state)

# hardcoded agent by hour
agent = HourAgent()
# agent = WeekdayAgent()

while not terminated:
    # agent is your own imported agent class
    action = agent.act(state)
    # action = np.random.uniform(-1, 1)
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = environment.step(action)
    # adds relevant features so that now each state is: 
    # [storage_level, price, hour, day, calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month]
    next_state = preprocess_state(next_state, timestamps)
    state = next_state
    aggregate_reward += reward
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)

print('Total reward:', aggregate_reward)