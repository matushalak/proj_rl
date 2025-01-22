from numpy import ndarray, append
from pandas import Series

def preprocess_state(state:ndarray, timestamps:Series) -> dict | ndarray:
    day = state[-1] - 1 # -1 to turn into index
    # useful features: calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month
    better_features = [timestamps[day].day, timestamps[day].weekday(), timestamps[day].weekofyear, timestamps[day].month]
    # all features: # [storage_level, price, hour, day, calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month]
    even_better = append(state, better_features)

    # dict with only features that are relevant

    """ 
    even_better = {'storage':better_state[0],
                   'price':better_state[1],
                   'hour':better_state[2],
                   'weekday':better_state[-3],
                   'week':better_state[-2],
                   'month':better_state[-1]}
    """
    return even_better # better_state


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