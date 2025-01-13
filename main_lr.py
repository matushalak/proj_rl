from env import DataCenterEnv
from utils import QLearningRegressor
import numpy as np
import argparse

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='train.xlsx')
    args = args.parse_args()

    np.set_printoptions(suppress=True, precision=2)
    path_to_dataset = args.path

    # Initialize environment and agent
    environment = DataCenterEnv(path_to_dataset)
    agent = QLearningRegressor()

    aggregate_reward = 0
    terminated = False
    state = environment.observation()
    
    # Training loop
    step_counter = 0
    print_frequency = 24 * 30  # Print monthly statistics
    
    # Episode statistics
    episode_rewards = []
    daily_rewards = 0
    
    while not terminated:
        storage_level = state[0]
        
        # Get action from agent using Q-value maximization
        action = agent.get_action(state, storage_level)
        
        # Execute action
        next_state, reward, terminated = environment.step(action)
        
        # Update Q-function using Bellman equation
        agent.update(state, action, reward, next_state)
        
        # Update statistics
        daily_rewards += reward
        aggregate_reward += reward
        step_counter += 1
        
        # Daily reset
        if step_counter % 24 == 0:
            episode_rewards.append(daily_rewards)
            daily_rewards = 0
            
        # Print statistics monthly
        if step_counter % print_frequency == 0:
            current_day = state[3]
            avg_daily_reward = np.mean(episode_rewards[-30:]) if len(episode_rewards) >= 30 else np.mean(episode_rewards)
            print(f"Day {current_day} | "
                  f"Monthly Avg Daily Reward: {avg_daily_reward:.2f} | "
                  f"Total Reward: {aggregate_reward:.2f}")
            
        state = next_state

    print('\nTraining Complete!')
    print(f'Final Total Reward: {aggregate_reward:.2f}')
    print(f'Average Daily Reward: {aggregate_reward/state[3]:.2f}')

if __name__ == "__main__":
    main()