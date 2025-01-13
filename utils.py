from numpy import ndarray, append
from pandas import Series

def preprocess_state(state:ndarray, timestamps:Series) -> ndarray:
    day = state[-1] - 1 # -1 to turn into index
    # useful features: calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month
    better_features = [timestamps[day].day, timestamps[day].weekday(), timestamps[day].weekofyear, timestamps[day].month]
    better_state = append(state, better_features)
    return better_state