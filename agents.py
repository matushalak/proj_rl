from numpy import ndarray, array
from numpy.random import uniform

# State -> Action
class HourAgent:
    def __init__(self):
        pass

    # State -> Action [-1 : +1]
    def act(self, state:dict|ndarray) -> float:
        hour, weekday, month = state['hour'], state['weekday'], state['month']
        
        # just simplest purely hour based
        if (1 <= hour and hour < 9) or hour > 23: 
                return 1 
            
        elif (10 <= hour and hour <= 14):
            return -1
        
        else:
            return uniform(-1, 1)

class WeekdayAgent:
    def __init__(self):
        pass
    
    # State -> Action [-1 : +1]
    def act(self, state:dict|ndarray) -> float:
        hour, weekday, month = state['hour'], state['weekday'], state['month']

        # weeek
        if weekday <= 4:    
            if (1 <= hour and hour < 9) or hour > 21: 
                    return uniform(0.6, 1)
                
            elif (10 <= hour and hour <= 14) or (18 <= hour and hour <= 21):
                return uniform(-1, -0.8)
            
            else:
                return uniform(-.65, .65)

       # weekend     
        else:
            if (1 <= hour and hour < 9) or hour > 21: 
                return 1
                
            elif (10 <= hour and hour <= 14) or (18 <= hour and hour <= 21):
                return uniform(-1, -0.6)
            
            else:
                return uniform(-.3, .65)
        

