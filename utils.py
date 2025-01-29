from numpy import ndarray, append
from pandas import Series
import re

def preprocess_state(state:ndarray, timestamps:Series) -> dict | ndarray:
    day = state[-1] - 1 # -1 to turn into index
    # useful features: calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month
    better_features = [timestamps[day].day, timestamps[day].weekday(), timestamps[day].weekofyear, timestamps[day].month]
    # all features: # [storage_level, price, hour, day, calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month]
    better_state = append(state, better_features)

    # dict with only features that are relevant
    even_better = {'storage':better_state[0],
                   'price':better_state[1],
                   'hour':better_state[2],
                   'weekday':better_state[-3],
                #    'week':better_state[-2],
                   'month':better_state[-1]}
    
    return even_better # better_state

def extract_bf_number(file_name):
        match = re.search(r"BF(-?\d+\.?\d*)", file_name)
        if match:
            return float(match.group(1))  # Convert the number to float
        return float('-inf')  # Return negative infinity if no match is found