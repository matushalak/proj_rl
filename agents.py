from collections import deque
from numpy import ndarray, array
from numpy.random import uniform, normal

#%% Baselines
class RandomAgent:
    def __init__(self):
        pass

    # State -> Action [-1 : +1]
    def act(self, state:dict|ndarray) -> float:
        return uniform(-1,1)

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

        if weekday >= 5:
            # just simplest purely hour based
            if (1 <= hour and hour < 9) or hour > 23: 
                    return 1 
                
            elif (10 <= hour and hour <= 14):
                return -1
            
            else:
                return uniform(-.2, 1)
        
        else:
             # just simplest purely hour based
            if (1 <= hour and hour < 9) or hour > 23: 
                    return 1 
                
            elif (10 <= hour and hour <= 14):
                return -1
            
            else:
                return uniform(-.75, .75)

class Average:
    def __init__(self, window_size:int = 25):
        # sliding window
        self.buffer = deque(maxlen=window_size)
        self.window_size = window_size
        self.moving_average = 0

    def act(self, state) -> float:
        hour, weekday, month, price, storage = state['hour'], state['weekday'], state['month'], state['price'], state['storage']
        # add do price buffer
        self.buffer.append(price)
        #update mva
        if len(self.buffer) < self.window_size:
            self.moving_average = sum(self.buffer) / len(self.buffer)
        else:
            self.moving_average = sum(self.buffer) / self.window_size

        if price < self.moving_average:
            # action = uniform(0, 1)
            action = 1
        elif price > self.moving_average:
            # action = uniform(-1, 0)
            action = -1
        else:
            action = 0

        # Scale action with the deviation (e.g., normalized by moving average)
        # Calculate deviation from moving average
        # deviation = price - self.moving_average
        # # doesnt work well...
        # if deviation > 0:  # Price above moving average (Sell)
        #     action = -min(deviation / self.moving_average, 1)  # Ensure action doesn't exceed -1
        # elif deviation < 0:  # Price below moving average (Buy)
        #     action = min(-deviation / self.moving_average, 1)  # Ensure action doesn't exceed +1
        # else:
        #     action = 0  # No deviation, no action

        return action

class AverageHour:
     # based on gridsearch
     def __init__(self, window_size:int = 25,
                  monthADD = 0,
                  PEAKweek = 0,
                  PEAKweekend = 0,
                  OFFPEAKweekSELL = 0,
                  OFFPEAKweekendSELL = 0,
                  OFFPEAKweekBUY = .25,
                  OFFPEAKweekendBUY = 1):
        # sliding window
        self.buffer = deque(maxlen=window_size)
        self.window_size = window_size
        self.moving_average = 0

        self.monthADD = monthADD
        self.PEAKweek = PEAKweek
        self.PEAKweekend = PEAKweekend
        self.OFFPEAKweekSELL = OFFPEAKweekSELL
        self.OFFPEAKweekendSELL = OFFPEAKweekendSELL
        self.OFFPEAKweekBUY = OFFPEAKweekBUY
        self.OFFPEAKweekendBUY = OFFPEAKweekendBUY


     def act(self, state:dict|ndarray) -> float:
        hour, weekday, month, price, storage = state['hour'], state['weekday'], state['month'], state['price'], state['storage']
          
        # add do price buffer
        self.buffer.append(price)
        #update mva
        if len(self.buffer) < self.window_size:
            self.moving_average = sum(self.buffer) / len(self.buffer)
        else:
            self.moving_average = sum(self.buffer) / self.window_size

        # Always buy here
        if (1 <= hour and hour < 9) or hour > 23: 
            return 1 
        
        # on weekdays sell, on weekends not so much
        elif (10 <= hour and hour <= 14):
            return self.PEAKweek if weekday < 5 else self.PEAKweekend
            # return normal(loc = -0.8) if weekday < 5 else normal(loc = -0.3)
        
        else:
            monthly_add = self.monthADD if (month > 1 and month < 9) else 0
            if price > self.moving_average:
                return self.OFFPEAKweekSELL + monthly_add if weekday < 5 else self.OFFPEAKweekendSELL + monthly_add
                # return normal(loc = -0.7)+ monthly_add if weekday < 5 else normal(loc = -.5)+ monthly_add
            
            elif price < self.moving_average:
                return self.OFFPEAKweekBUY + monthly_add if weekday < 5 else self.OFFPEAKweekendBUY + monthly_add
            
            else:
                return 0
            

#%% Tabular RL
class QAgent:
    def __init__(self):
        # TODO: check discretization choices
        self.state_dims = (17) 
    
    def train(self):
        # save Q-table, if not present trigger train, otherwise just read Q-table & update it
        pass

    def test(self):
        pass
    
    def update(self):
        pass

    def act(self, state) -> float:
        pass

          
