from numpy import ndarray
from numpy.random import uniform

# State -> Action
class HourAgent:
    def __init__(self):
        pass

    # State -> Action [-1 : +1]
    def act(self, state:dict|ndarray) -> float:
        # just simplest purely hour based
        if 0 <= state['hour'] and state['hour'] <= 9:
            return 1 
        
        elif (10 <= state['hour'] and state['hour'] <= 13):
            return -1
        
        else:
            return uniform(-1, 1)
        

