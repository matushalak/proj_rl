from numpy import ndarray, append
from pandas import Series

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