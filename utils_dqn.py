import numpy as np
from pandas import Series

def shape_reward(state, action, reward, x = 2):
    storagelvl = state[0]
    price = state[1]
    hour = state[2]


    if hour == 24:
        reward -= action * storagelvl

    if (storagelvl - 50) / (24 - hour) > 10 and action >= 0:
        reward -= 100 * x

    if (storagelvl - 50) / (24 - hour) > 10 and action < 0:
        reward += 100 * x

    if price > 0.4 and action < 0:
        reward += price * 100 * x

    if action < 0:
        reward -= abs(action * 10)

    if action >= 0:
        reward += abs(action * 10)


    return reward


def preprocess_state(state:np.ndarray, timestamps:Series) -> np.ndarray:
    # Normalize storage_level and price
    storagelvl = state[0]
    price = state[1]
    
    normalized_storage = storagelvl / 220.0 if storagelvl <= 220 else 1.0
    normalized_price = price / 1000.0 if price <= 1000 else 1.0

    better_features = [normalized_storage, normalized_price]
   
    hour = state[2]
    day = state[3] - 1 # -1 to turn into index
    date = timestamps[day]
    weekday = date.weekday() + 1
    week = date.weekofyear

     # Daily cycle (hour of the day)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Weekly cycle (day of the week)
    day_sin = np.sin(2 * np.pi * weekday / 7)
    day_cos = np.cos(2 * np.pi * weekday / 7)
    
    # Yearly cycle (week of the year)
    week_sin = np.sin(2 * np.pi * week / 52)
    week_cos = np.cos(2 * np.pi * week / 52)

    features = [hour_sin, hour_cos, day_sin, day_cos, week_sin, week_cos]

    for feature in features:
        better_features.append((feature+1)/2)

    assert len(better_features) == 8

    return better_features # better_state


def discretize_actions(action):
    if -1 <= action < -0.5:
        return 0
    elif -0.5 <= action < 0:
        return 1
    elif action == 0:
        return 2
    elif 0 < action < 0.5:
        return 3
    elif 0.5 <= action <= 1:
        return 4

def normalize_actions(action):
    if action == 0:
        return -1
    elif action == 1:
        return -0.5
    elif action == 2:
        return 0
    elif action == 3:
        return 0.5
    elif action == 4:
        return 1