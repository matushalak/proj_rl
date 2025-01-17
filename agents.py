from collections import deque
from numpy import ndarray, array, linspace, argmax, zeros, interp, digitize
from numpy.random import uniform, normal
from pandas import DataFrame, merge
from env import DataCenterEnv
from utils import preprocess_state

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
# epsilon greedy
# or USB rule 
class QAgent:
    def __init__(self):
        # moving averages
        self.mvaD = deque(maxlen=25) # 1 day
        self.mva2D = deque(maxlen=49) # 2 days
        self.mvaW = deque(maxlen=24*7 + 1) # 1 week

        # TODO: check discretization choices
        # Hour (12), Storage level (17), Action (9) [by 0.25], Price Above / Below MVA (2), Weekend (2), Winter (2)
        # Start simple
        # Hour (12), Storage (17), Action (9)
        self.hours = linspace(2, 24, 12)
        self.storage = linspace(10,170, 17)
        self.actions = linspace(-1,1, 9)

        self.state_dims = (self.hours.size, self.storage.size, self.actions.size) 

        # Qtable, initialize with zeros
        self.Qtable = zeros(self.state_dims)
        
        # Epsilon
        # after 5000 episodes, exploit
        self.epsilon_decay = 5000
        # range of epsilon
        # start always exploring  100 % time end up exploring only 5 % of time
        self.epsilon_range = [1, .05] 

    def discretize_state(self, state):
        # 'storage': 80.0, 'price': 25.48, 'hour': 9.0, 'weekday': 5.0, 'month': 12.0}
        pass
    
    def experience_buffer(self, ENV: DataCenterEnv, episode_length:int) -> list:
        # 1) Modify features
        DateInfo = ENV.timestamps

        melted_data = ENV.test_data.melt(id_vars='PRICES',
                                        var_name= 'Hour',
                                        value_name='Price')
        melted_data.rename(columns={'PRICES':'Date'}, inplace=True)

        melted_data['Hour'] = melted_data['Hour'].str.replace('Hour ', '').astype(int)

        features_df = DataFrame({'Date':DateInfo,
                                'Weekend':DateInfo.apply(lambda x: 0 if x.weekday() < 5 else 1), # WEEK 0, WEEKEND 1 Mon {0} - Sun {6}
                                'Winter':DateInfo.apply(lambda x: 0 if x.month not in (10, 11, 12) else 1)}) # Jan {1} - Dec {12}
        
        Features = merge(melted_data, features_df, on = 'Date', how = 'left')
        
        # 2) Sample episodes
        episodes = []
        # loop through all timepoints and 
        episode = deque(maxlen=episode_length)
        
        pass

    # reward shaping
    def train(self, dataset) -> ndarray:
        # training environment
        env = DataCenterEnv(dataset)
        
        # experience buffer
        self.experience_buffer

        # save Q-table, if not present trigger train, otherwise just read Q-table & update it
        
        pass
    
    def update(self):
        pass

    def act(self, state) -> float:
        pass

          
