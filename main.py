from env import DataCenterEnv
import numpy as np
import argparse
from utils import preprocess_state
from agents import HourAgent, WeekdayAgent, Average, AverageHour, QAgent
import os
import re

def main(path_to_dataset:str, retrain:bool = False, PRINT:bool = False, agent_params:list|bool = False, retACTIONS: bool = False, 
         Agent:object = QAgent) -> float:
    # 1) Prepare / train agent
    # hardcoded agent by hour
    # if agent_params:
    #     agent = AverageHour(*agent_params)
    # else:
    #     agent = AverageHour()
    # agent = Average()
    
    # agent = HourAgent()
    # agent = WeekdayAgent()
    # Function to extract the BF (best fitness) number
    def extract_bf_number(file_name):
        match = re.search(r"BF(-?\d+\.?\d*)", file_name)
        if match:
            return float(match.group(1))  # Convert the number to float
        return float('-inf')  # Return negative infinity if no match is found

    if Agent == QAgent:
        Qtables = [file for file in os.listdir() if file.startswith('Qtable')]
        if len(Qtables) == 0 or retrain == True:
            agent = Agent()
            QTABLE = agent.train(dataset = 'train.xlsx')
        
        else:
            # Find the file with the highest BF number
            highest_bf_file = max(Qtables, key=extract_bf_number)
            print(f'Using this Qtable: {highest_bf_file}')
            agent = Agent(Qtable_dir = highest_bf_file)
            
    
    else:
        agent = Agent()
        
    # 2) run agent on dataset
    environment = DataCenterEnv(path_to_dataset)
    # dates
    timestamps = environment.timestamps

    aggregate_reward = 0
    terminated = False
    state = environment.observation()
    # adds relevant features so now each state is: 
    # [storage_level, price, hour, day, calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month]
    state = preprocess_state(state, timestamps)
    if PRINT:
        print("Starting state:", state)

    actions = []
    hour = 0

    while (not terminated) or (hour != 24):
        # agent is your own imported agent class
        action = agent.act(state)

        actions.append(action)
        # action = np.random.uniform(-1, 1)
        # next_state is given as: [storage_level, price, hour, day]
        next_state, reward, terminated = environment.step(action)
        # adds relevant features so that now each state is: 
        # [storage_level, price, hour, day, calendarday (1 - 31st), weekday (Mon{0} - Sun{6}), week, month]
        next_state = preprocess_state(next_state, timestamps)
        hour = next_state['hour']
        state = next_state
        aggregate_reward += reward
        if PRINT:
            print("Action:", action)
            print("Next state:", next_state)
            print("Reward:", reward)

    nyears = len(timestamps) // 365 
    
    # if PRINT:
    print(f'Total reward in {nyears} years:', aggregate_reward)
    print('Average reward / year', aggregate_reward / nyears)

    if retACTIONS:
        return aggregate_reward / nyears, actions
    else:
        return aggregate_reward / nyears

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='validate.xlsx')
    args.add_argument('--retrain', type=bool, default=False)
    args = args.parse_args()

    np.set_printoptions(suppress=True, precision=2)
    path_to_dataset = args.path
    reTRAIN = args.retrain
    
    main(path_to_dataset=path_to_dataset, retrain=reTRAIN)