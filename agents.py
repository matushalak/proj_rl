from collections import deque
from itertools import product
from numpy import ndarray, array, linspace, argmax, arange, zeros, interp, digitize, save, load, mean, std, count_nonzero, where
from numpy import abs as absolute
from numpy.random import uniform, choice
from pandas import DataFrame, merge, Series
from env import DataCenterEnv
from utils import preprocess_state
import os
import matplotlib.pyplot as plt

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
    def __init__(self, discount_rate = .99, Qtable_dir:str = False): 
        # discount rate was .99 for most of the time during testing (need really high DF - even .9 didnt work)
        self.needed_storage = 120

        # Discount rate
        self.discount_rate = discount_rate

        # Learning rate
        self.learning_rate_range = [0.1, 0.001] # big fluctuations in environment, use SMALL  learning rate
        self.lr_decay = 200

        # Epsilon
        # after 2000 - 5000 episodes, exploit
        self.epsilon_decay = 100
        # range of epsilon
        # start always exploring  100 % time end up exploring only 5 - 10 % of time
        self.epsilon_range = [1, .001] 

        # moving averages
        if Qtable_dir:
            self.daily_price = deque(maxlen=25) # 1 day
            self.price2D = deque(maxlen=49) # 2 days
            self.weekly_price = deque(maxlen=24*7 + 1) # 1 week

        # TODO: check discretization choices ~
        # Hour (12), Storage level (17), Action (9) [by 0.25], Price Above / Below MVA (2), Weekend (2), Winter (2)
        # Start simple
        # Hour (12), Storage (17), Action (9)
        

        # morning (1-9), lunchtime (10 - 13),  afternoon (14 - 17), evening (18 - 21), night (22 - 24)
        self.hours = array([9,14,18,22])
        self.storage = array([40, 80, 120, 150])#linspace(10,170, 17) # [40, 80, 120, 150] # [20, 40, 60, 80, 100, 120, 150]
        self.actions = array([-1,0,1])
        
        # more features
        self.winter = array([0,1])
        self.weekend = array([0,1])

        # above x std, below x std, within x std
        self.above_daily = array([0,1,2])
        self.above_2D = array([0,1, 2])
        self.above_weekly = array([0,1, 2])

        # 5 x 18 x 2 x 3
        self.state_dims = (self.hours.size + 1, self.storage.size + 1, self.above_daily.size, 
                            self.actions.size) #self.winter.size, self.weekend.size,

        # Qtable, initialize with zeros
        if Qtable_dir:
            self.Qtable = load(Qtable_dir)
            # breakpoint()
        else:
            self.Qtable = zeros(self.state_dims)
            self.smart_initialize()

            assert self.Qtable.size - count_nonzero(self.Qtable) == 0

        self.state_visits = zeros(self.Qtable.shape)
            # check for uncaught elements in smart initialization
            # h, s,  p, a = where(self.Qtable == 0)
            # for sts in zip(h,s,p,a):
            #     print(sts)
            # breakpoint()


    def discretize_state(self, state:dict):
        # eg. state 'storage': 80.0, 'price': 25.48, 'hour': 9.0, 'weekday': 5.0, 'month': 12.0}
        # breakpoint()
        storage = digitize(state['storage'], self.storage) # discretized to 18 bins
        hour = digitize(state['hour'], self.hours) # discretized to 5 bins
        weekend = 0 if state['weekday'] < 5 else 1
        winter = 0 if state['month'] not in (10, 11, 12) else 1
        # potentially add above / below moving average

        # want to return just the indices, since we will use this to index the Q table
        # return [storage, state_dict['price'], hour, weekend, winter]
        # 2 days
        diff_p = mean(self.price2D) - state['price']
        daily_p = 1 if diff_p >= 1.5* std(self.price2D) else (0 if diff_p <= -1.5 * std(self.price2D) else 2)
        
        # 1 day
        # diff_p = mean(self.daily_price) - state['price']
        # daily_p = 1 if diff_p >= 1.5* std(self.daily_price) else (0 if diff_p <= -1.5 * std(self.daily_price) else 2)
        
        # int, int, bool
        return [hour, storage, daily_p] # for now winter, weekend
    
    # ~TODO: possibly incorporate weekend / winter info here
    def smart_initialize(self):
        for h, stor, pr, act in product(arange(1, 25), arange(0, 180, 10), self.above_2D, self.actions): # self.winter, self.weekend,
            hi = digitize(h, self.hours)
            si = digitize(stor, self.storage)
            pi = list(self.above_2D).index(pr)
            ai = list(self.actions).index(act)

            indices = (hi, si, pi, ai) # wi, we, 

            # urgency = (25 - h) / 24
            # self.hours = array([9,14,18,22])
            debt = self.needed_storage - stor
            hours_left = 24 - h

            # winter (more conservative), summer (cheaper)
            winterBUY = 0 #200 if wi == 0 else -100
            winterSELL = 0 #100 if wi == 0 else 150

            # weekend (cheaper, buy more)
            weekendBUY = 0 #-75 if we == 0 else 125
            weekendSELL = 0 #125 if we == 0 else -75

            if debt > (hours_left * 10):
                # print(h, stor, pr, act)
                # really bad, want to avoid at all costs
                self.Qtable[indices] = - 10000
            
            # less than 120 storage
            else:
                if stor < self.needed_storage:    
                    if act == -1:
                        self.Qtable[indices] = - 500 #+ winterSELL + weekendSELL
                    elif act == 0:
                        self.Qtable[indices] = - 200
                    else:
                        self.Qtable[indices] = 500 #+ winterBUY + weekendBUY
                
                # above storage capacity
                else:
                    if act == -1:
                        self.Qtable[indices] = 500 #+ winterSELL + weekendSELL
                    elif act == 0:
                        self.Qtable[indices] = -50
                    else:
                        self.Qtable[indices] = -500 #+ winterBUY + weekendBUY
            
            # print(h, stor, pr, act)
            # print(self.Qtable[hi, si, pi, ai])
        

    # TODO 1: reward shaping ~
    def train(self, dataset:str, simulations:int = 300) -> ndarray:
        # for eps 100
        # for eps 200 - lr 400, episodes 500
        # for eps 300 - lr 450, episodes 600
        ''''
        Initial (intended) version: 
            -> View the whole train dataset as 1 long episode
        
        Better performance ?: 
            -> EPISODES learning (experience buffer etc.)
            -> reward shaping
        '''
        print(f'Starting training for {simulations} iterations through the dataset!') 
        best_QT = self.Qtable.copy()

        # training environment
        env = DataCenterEnv(dataset)
        tstamps = env.timestamps
        
        terminated = False
        total_rewards = []
        training_costs = []
        validation_costs = []
        # save Q-table, if not present trigger train, otherwise just read Q-table & update it
        
        # key difference between STATE and 
        # NEXT state (this we get from env.step(action)) {but dont know future PRICE!}
        # and we use thata to update the STATE
        
        # number of passes through dataset
        # just take Q-learning from tutorial
        for isim in range(simulations + 1):
            # reset every pass through dataset 
            self.daily_price = deque(maxlen=25) # 1 day
            self.price2D = deque(maxlen=49) # 2 days
            self.weekly_price = deque(maxlen=24*7 + 1) # 1 week
            
            # for now, reset to beginning of whole dataset
            state = preprocess_state(
                env.reset(), tstamps)
            state_i = self.discretize_state(state)

            # decaying learning rate
            LR = interp(isim, [0, self.lr_decay], self.learning_rate_range)

            terminated = False
            total_reward = 0
            # decaying adaptive epsilon (initially)
            self.epsilon = interp(isim, [0, self.epsilon_decay], self.epsilon_range)
            
            print(f"The current epsilon rate is {self.epsilon}, Learning rate is {LR}")
            
            while (not terminated) or (state['hour'] != 24):
                self.daily_price.append(state['price'])
                self.price2D.append(state['price'])
                self.weekly_price.append(state['price'])
                
                # I) Action is index of Qtable action dimension
                if uniform() < self.epsilon:
                    # epsilon greedy
                    action = choice(self.actions)
                    action = list(self.actions).index(action)

                else:
                    # best action (lowest cost = max value)
                    action = argmax(self.Qtable[state_i[0], state_i[1], state_i[2], :]) # state_i[3], state_i[4],
                
                self.state_visits[(*state_i, action)] += 1

                # II) Take step according to action
                next_state, reward , terminated = env.step(self.actions[action])
                # REWARD SHAPING
                if state['storage'] <= 120:
                    if self.actions[action] == 1:  # Incentivize buying when storage is low
                        RS = 1/3 * abs(reward)
                    elif self.actions[action] == -1:  # Penalize selling when storage is low
                        RS = -1/2 * abs(reward)
                    else: # penalize not doing anything
                        RS = -1/5 * abs(reward)

                elif state['storage'] > 130:
                    if self.actions[action] == -1:  # Reward selling 
                        RS = 1/3* abs(reward)
                    elif self.actions[action] == 1:  # Penalize unnecessary buying
                        RS = -1/2 * abs(reward)
                    else: #penalize not doing anything
                        RS = -1/3 * abs(reward)

                    
                # RS_TERM = # 1* state['storage'] / (25 - state['hour'])

                next_state = preprocess_state(next_state, tstamps)
                next_state_i = self.discretize_state(next_state) 

                # TD learning (umbrella term including Q-learning), just a way to break up Q-formula into 2 steps:
                # 1) Immediate reward + discounted max {future reward}
                target = reward + RS + self.discount_rate * max(self.Qtable[next_state_i[0], next_state_i[1], next_state_i[2],:]) # next_state_i[3], next_state_i[4],
                # 2) Difference from current Q-value of state and what's possible if taking best action
                td_error = target - self.Qtable[state_i[0], state_i[1], state_i[2],action] # state_i[3], state_i[4],
                # 3) Update Q-value for given state towards the optimal Q-value by adding td_error proportional to learning rate
                self.Qtable[state_i[0], state_i[1], state_i[2], action] = self.Qtable[state_i[0], state_i[1], state_i[2],action
                                                                          ] + LR * td_error
                
                # DEbugging
                # print(f'State {state}, indices {state_i}, action {self.actions[action]}, index {action}')
                # print(f'Next State {next_state}, indices {next_state_i}')
                
                state, state_i = next_state, next_state_i
                total_reward += reward
            # yearly
            print(f'Round {isim}: Total training Cost = {total_reward}, Unvisited States: {self.Qtable.size - count_nonzero(self.state_visits)}')    
            total_rewards.append(total_reward / (len(tstamps) // 365))
            
            if isim % 50 == 0:
                tr = self.test('train.xlsx')
                val = self.test('validate.xlsx')
                training_costs.append(tr)
                validation_costs.append(val)

                print(f"Total REAL: (training) cost = {tr}; validation cost = {val}")
                

                if isim > 0:
                    plt.figure()
                    plt.plot(arange(isim+1), total_rewards, label = 'training + reward shaping', color = 'r')
                    plt.plot(arange(stop = isim + 1, step = 50), training_costs, label = 'training REAL' ,color = 'b')
                    plt.plot(arange(stop = isim + 1, step = 50), validation_costs, label = 'validation REAL', color= 'g')
                    plt.legend(loc = 'upper left')
                    plt.xlabel('Iterations through dataset')
                    plt.ylabel('Cost')
                    plt.tight_layout()
                    plt.savefig(f'after{isim}_epsd{self.epsilon_decay}_lrd{self.lr_decay}_df{self.discount_rate}.png') # WkWi_inRS
                    plt.close()

                    if mean([tr, val]) >= best_fit:
                        best_QT = self.Qtable.copy()
                        best_fit = mean([tr, val])
                
                elif isim == 0: # first pass
                    best_fit = mean([tr, val])

        print('Training done!')

        save(f'Qtable_{simulations}sims_epsd{self.epsilon_decay}lrd{self.lr_decay}_df{self.discount_rate}_BF{round(best_fit / 1e6, 3)}.npy', best_QT) # WkWi_inRS-
        
        return best_QT


    def test(self, dataset:str) -> float:
        # reset every pass through dataset 
        self.daily_price = deque(maxlen=25) # 1 day
        self.price2D = deque(maxlen=49) # 2 days
        self.weekly_price = deque(maxlen=24*7 + 1) # 1 week
        # 2) run agent on dataset
        TESTenvironment = DataCenterEnv(dataset)
        # dates
        timestamps = TESTenvironment.timestamps

        aggregate_reward = 0
        terminated = False
        S = TESTenvironment.reset()
        # adds relevant features so now each state is: 
        # [storage_level, price, hour, day, calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month]
        S = preprocess_state(S, timestamps)

        actions = []
        hour = 0

        while (not terminated) or (hour != 24):
            # agent is your own imported agent class
            action = self.act(S)

            actions.append(action)
            # action = np.random.uniform(-1, 1)
            # next_state is given as: [storage_level, price, hour, day]
            NS, reward, terminated = TESTenvironment.step(action)
            # adds relevant features so that now each state is: 
            # [storage_level, price, hour, day, calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month]
            NS = preprocess_state(NS, timestamps)
            hour = NS['hour']
            S = NS
            aggregate_reward += reward

        nyears = len(timestamps) // 365 
        
        return aggregate_reward / nyears


    def act(self, state) -> float:
        self.daily_price.append(state['price'])
        self.price2D.append(state['price'])
        self.weekly_price.append(state['price'])
        
        state = self.discretize_state(state)
        # argmax on action dimension
        # breakpoint()
        action = argmax(self.Qtable[state[0], state[1], state[2],:]) #state[3], state[4],
        # breakpoint()
        return self.actions[action]
    


if __name__ == '__main__':
    ag = QAgent()
    ag.experience_buffer(env)
    env = DataCenterEnv('train.xlsx')