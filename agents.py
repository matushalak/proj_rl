from collections import deque
from numpy import ndarray, array, linspace, argmax, argmin, zeros, interp, digitize, save
from numpy import abs as absolute
from numpy.random import uniform, choice
from pandas import DataFrame, merge, Series
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
    def __init__(self, discount_rate = .95):
        # Discount rate
        self.discount_rate = discount_rate

        # Learning rate
        self.learning_rate = 0.1 # ???

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
        # after 2000 - 5000 episodes, exploit
        self.epsilon_decay = 1000
        # range of epsilon
        # start always exploring  100 % time end up exploring only 5 - 10 % of time
        self.epsilon_range = [1, .1] 

    def discretize_state(self, state:dict):
        # 'storage': 80.0, 'price': 25.48, 'hour': 9.0, 'weekday': 5.0, 'month': 12.0}
        storage = argmin(absolute(self.storage - state['storage']))  #digitize(state['storage'], self.storage)
        hour = argmin(absolute(self.hours - state['hour'])) #digitize(state['hour'], self.hours)
        weekend = 0 if state['weekday'] < 5 else 1
        winter = 0 if state['month'] not in (10, 11, 12) else 1
        # potentially add above / below moving average

        # want to return just the indices, since we will use this to index the Q table
        # return [storage, state_dict['price'], hour, weekend, winter]
        return [hour, storage, state['price']] # for now
    
    # better to transition from short to long episodes
    # first try without and view dataset as one long episode 
    # later try to work around to modify ENV attributes after initialization && reset after episode is over
    def experience_buffer(self, ENV: DataCenterEnv, episode_length:int = 24) -> list:
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
        Features = Features.sort_values(by=['Date', 'Hour']).reset_index(drop=True)
        
        # 2) Sample episodes
        episodes = []
        # loop through all timepoints and 
        for i in range(0, len(Features) - episode_length + episode_length // 2, episode_length // 2):
            episodes.append(Features.iloc[i : i + episode_length + 1, -4:])
        breakpoint()
        

    # TODO 1: reward shaping
    # TODO 2: experience buffer & transition from short to long-term strategies
    def train(self, dataset:str, simulations:int = 2000) -> ndarray:
        ''''
        Initial (intended) version: 
            -> View the whole train dataset as 1 long episode
        
        Better performance ?: 
            -> EPISODES learning (experience buffer etc.)
            -> reward shaping
        '''
        print(f'Starting training for {simulations} iterations through the dataset!')
        
        # training environment
        env = DataCenterEnv(dataset)
        tstamps = env.timestamps
        
        terminated = False
        total_rewards = []
        # save Q-table, if not present trigger train, otherwise just read Q-table & update it
        
        # key difference between STATE and 
        # NEXT state (this we get from env.step(action)) {but dont know future PRICE!}
        # and we use thata to update the STATE
        
        # number of passes through dataset
        # just take Q-learning from tutorial
        for isim in range(simulations):
            # for now, reset to beginning of whole dataset
            state = preprocess_state(
                env.reset(), tstamps)
            state_i = self.discretize_state(state)

            terminated = False
            total_reward = 0
            # decaying adaptive epsilon (initially)
            self.epsilon = interp(isim, [0, self.epsilon_decay], self.epsilon_range)
            
            print(f"The current epsilon rate is {self.epsilon}")
            
            while (not terminated) or (state['hour'] != 24):
                # I) Action is index of Qtable action dimension
                if uniform() < self.epsilon:
                    # epsilon greedy
                    action = choice(self.actions)
                    action = argmin(absolute(self.actions - action))

                else:
                    # best action (lowest cost = max value)
                    action = argmax(self.Qtable[state_i[0], state_i[1], :])
                
                # II) Take step according to action
                next_state, reward , terminated = env.step(self.actions[action])

                next_state = preprocess_state(next_state, tstamps)
                next_state_i = self.discretize_state(next_state) 

                # TD learning (umbrella term including Q-learning), just a way to break up Q-formula into 2 steps:

                # 1) Immediate reward + discounted max {future reward}
                target = reward + self.discount_rate * max(self.Qtable[next_state_i[0], next_state_i[1],:])
                # 2) Difference from current Q-value of state and what's possible if taking best action
                td_error = target - self.Qtable[state_i[0], state_i[1], action]
                # 3) Update Q-value for given state towards the optimal Q-value by adding td_error proportional to learning rate
                self.Qtable[state_i[0], state_i[1], action] = self.Qtable[state_i[0], state_i[1], action
                                                                          ] + self.learning_rate * td_error

                state, state_i = next_state, next_state_i
                total_reward += reward
            print(f'Round {isim}: Total Cost = {total_reward}')
            total_rewards.append(total_reward)
        print('Training done!')
        save(f'Qtable_eps_decay{self.epsilon_decay}.npy', self.Qtable)

    def act(self, state) -> float:
        state = self.discretize_state(state)
        # argmax on action dimension
        action = argmax(self.Qtable[state[0], state[1], :])
        return self.actions[action]

          
if __name__ == '__main__':
    ag = QAgent()
    env = DataCenterEnv('train.xlsx')
    ag.experience_buffer(env)