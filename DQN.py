from collections import deque

from env import DataCenterEnv
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils_dqn import *
from cma import CMAEvolutionStrategy
from scipy.special import softmax
import random
import matplotlib.pyplot as plt

seed = 33
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

episode_length = 1096 * 24

class DQN(nn.Module):
    def __init__(self, state_size, action_size, lr):
        super(DQN, self).__init__()

        self.dense1 = nn.Linear(state_size, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        def init_weights(m):
            # if type(m) == nn.Linear:
            #     torch.nn.init.xavier_uniform_(m.weight)
            #     m.bias.data.fill_(0.01)
            if type(m) == nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(init_weights) # Apply the initialization

    def forward(self, state):
        x = F.relu(self.dense1(state))  # ReLU activation
        x = F.relu(self.dense2(x))      # ReLU activation
        x = F.relu(self.dense3(x))      # ReLU activation
        x = self.dense4(x)              # Output layer (no activation)
        return x
    
class PrioritizedReplay:
    def __init__(self, env, buffer_size, min_replay_size = 1000, alpha = 0.75, beta = 0.8, seed=seed):
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([-7000000.0], maxlen = 100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.beta = beta

        timestamps = self.env.timestamps
        state = preprocess_state(self.env.observation(), timestamps)

        for _ in range(self.min_replay_size):
            action = discretize_actions(np.random.uniform(-1, 1))
            next_state, reward, terminated = env.step(action)
            next_state = preprocess_state(next_state, timestamps)
            reward = shape_reward(state, action, reward)
            td_error = reward + 1e-5
            transition = (state, action, reward, terminated, next_state, td_error)
            self.replay_buffer.append(transition)
            state = next_state          

            if terminated:
                terminated = False
                state = preprocess_state(env.observation(), timestamps)

        print('init with random transitions done!')

    def add_data(self, data):
        self.replay_buffer.append(data)

    def sample(self, batch_size):
        priorities = [abs(transition[-1] + 1e-5)**self.alpha for transition in self.replay_buffer]
        probabilities = priorities / np.sum(priorities)
        sample_indices = np.random.choice(
            range(len(self.replay_buffer)), size=batch_size, p=probabilities)
        
        transitions = [self.replay_buffer[i] for i in sample_indices]

        weights = (len(self.replay_buffer) * probabilities[sample_indices]) ** (-self.beta)
        weights /= weights.max()

        #solutions
        states = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        next_states = np.asarray([t[4] for t in transitions])

        #transform to tensor
        states_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.float32).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.as_tensor(next_states, device=self.device, dtype=torch.float32)

        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)

        return states_t, actions_t, rewards_t, dones_t, next_states_t, weights, sample_indices

    def add_reward(self, reward):
        self.reward_buffer.append(reward)

    def update_priorities(self, indices, td_errors):

        for idx, td_error in zip(indices, td_errors):
            state, action, reward, terminated, next_state, old_td_error = self.replay_buffer[idx]
            updated_td_error = (abs(td_error) + 1e-5) ** self.alpha
            new_transition = state, action, reward, terminated, next_state, updated_td_error
            self.replay_buffer[idx] = new_transition

class DQNAgent:
    def __init__(self, env, device, epsilon_decay,
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed=seed):
        self.env = env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.lr = lr
        self.buffer_size = buffer_size

        self.replay_memory = PrioritizedReplay(self.env, self.buffer_size, seed=seed)
        self.online_net = DQN(8, 5, lr).to(self.device)

    def choose_action(self, step, state, greedy=False):

        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        random_sample = random.random()

        if (random_sample <= epsilon) and not greedy:
            action = discretize_actions(np.random.uniform(-1, 1))
        else:

            state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            q_vals = self.online_net(state.unsqueeze(0))

            max_q_index = torch.argmax(q_vals, dim=1)[0]
            action = max_q_index.detach().item()

        return action, epsilon

  #  def learn(self, batch_size):

        states, actions, rewards, dones, next_states = self.replay_memory.sample(batch_size)

        target_q_values = self.online_net(next_states)
        max_target_q_val = target_q_values.max(dim=1, keepdim=True)[0]

        target = rewards + self.discount_rate * max_target_q_val * (1 - dones)

        #loss
        q_values = self.online_net(states)
        #print(actions)
        action_q_values = torch.gather(q_values, 1, actions.long())

        loss = F.smooth_l1_loss(action_q_values, target.detach())

        self.online_net.optimizer.zero_grad()
        loss.backward()
        self.online_net.optimizer.step()

class DDQNAgent:
    def __init__(self, env, device, epsilon_decay,
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed=seed):
        self.env = env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.lr = lr
        self.buffer_size = buffer_size

        self.replay_memory = PrioritizedReplay(self.env, self.buffer_size, seed=seed)
        self.online_net = DQN(8, 5, lr).to(self.device)

        self.target_net = DQN(8, 5, lr).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())


    def choose_action(self, step, state, greedy=False):

        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        random_sample = random.random()

        if (random_sample <= epsilon) and not greedy:
            action = discretize_actions(np.random.uniform(-1, 1))
        else:

            state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            q_vals = self.online_net(state.unsqueeze(0))

            max_q_index = torch.argmax(q_vals, dim=1)[0]
            action = max_q_index.detach().item()

        return action, epsilon

    def return_q_value(self, observation):
        
        #We will need this function later for plotting the 3D graph
        
        obs_t = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
        q_values = self.online_net(obs_t.unsqueeze(0))
        
        return torch.max(q_values).item()

    def learn(self, batch_size):

        states, actions, rewards, dones, next_states, weights, indices = self.replay_memory.sample(batch_size)

        target_q_values = self.target_net(next_states)
        max_target_q_val = target_q_values.max(dim=1, keepdim=True)[0]

        target = rewards + self.discount_rate * max_target_q_val * (1 - dones)

        #loss
        q_values = self.online_net(states)
        #print(actions)
        action_q_values = torch.gather(q_values, 1, actions.long())

        loss = F.smooth_l1_loss(action_q_values, target.detach())
        weighted_loss = (weights * loss).mean()
        self.online_net.optimizer.zero_grad()
        weighted_loss.backward()
        self.online_net.optimizer.step()

        td_errors = (target - action_q_values).detach()
        self.replay_memory.update_priorities(indices, td_errors.cpu().numpy().flatten().tolist())

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())



# Hyperparameters
discount_rate = 0.99
batch_size = 32
buffer_size = 300000
epsilon_start = 0.1
epsilon_end = 0.01
epsilon_decay = 10000
max_steps = episode_length * 3

lr = 3e-4
target_update_frequency = 256

# TODO 
# dynamic discount rate

discount_rates = [random.uniform(0.9, 1) for _ in range(3)]
epsilon_starts = [random.uniform(0.1, 0.9) for _ in range(3)]
epsilon_decays = [random.uniform(1000, 10000) for _ in range(3)]
lrs = [random.uniform(0.0001, 0.001) for _ in range(3)]

env  = DataCenterEnv("train.xlsx")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training(env, agent, max_steps, target_ = False, seed = seed):
    aggregate_reward = 0
    avg_reward = []
    terminated = False
    state = preprocess_state(env.observation(), env.timestamps)

    for step in range(max_steps):

        action, epsilon = agent.choose_action(step, state)

        next_state, reward, terminated = env.step(normalize_actions(action))
        next_state = preprocess_state(next_state, env.timestamps)
        reward = shape_reward(state, action, reward)
        td_error = reward + 1e-5
        transition = (state, action, reward, terminated, next_state, td_error)
        agent.replay_memory.add_data(transition)
        state = next_state
        aggregate_reward += reward

        if terminated:
            print(f"Environment terminated at day {env.day}, hour {env.hour}")
            # env = DataCenterEnv("train.xlsx")
            env.reset()
            state = preprocess_state(env.observation(), env.timestamps)
            env = env
            print(f"Episode Reward: {aggregate_reward}")
            agent.replay_memory.add_reward(aggregate_reward)
            aggregate_reward = 0

        if step % 8 == 0:
            agent.learn(batch_size)

        if (step+1) % episode_length == 0:
            avg_reward.append(np.mean(agent.replay_memory.reward_buffer))

        if target_ and step % target_update_frequency == 0:
                ddqn_agent.update_target_net()

        if (step+1) % 10000 == 0:
            print(20 * '--')
            print('Step', step + 1)
            print('Epsilon', epsilon)
            #print('Avg Rew', np.mean(agent.replay_memory.reward_buffer))
            #print(avg_reward)
            print()

    return avg_reward

import itertools
all_combinations = list(itertools.product(
    discount_rates,
    epsilon_starts,
    epsilon_decays,
    lrs
))

# Print each combination
for i, (discount_rate, epsilon_start, epsilon_decay, lr) in enumerate(all_combinations, 1):
    print(f"\nCombination {i}:")
    print(f"Discount Rate: {discount_rate:.4f}")
    print(f"Epsilon Start: {epsilon_start:.4f}")
    print(f"Epsilon Decay: {epsilon_decay:.1f}")
    print(f"Learning Rate: {lr:.6f}")

    ddqn_agent = DDQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size)

    avg_rew_ddqn = training(env, ddqn_agent, max_steps, target_=True)
    print(avg_rew_ddqn, '\n', np.mean(avg_rew_ddqn), '\n', max(avg_rew_ddqn))

    torch.save(ddqn_agent.online_net.state_dict(), 'nn_state_dict_c' + str(i) + '.pt')


    config = {
        'seed': seed,
        'max_reward': max(avg_rew_ddqn),
        'discount_rate': discount_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
        'max_steps': max_steps,
        'learning_rate': lr,
        'target_update_frequency': target_update_frequency
    }

    # Write to file with nice formatting
    with open('log.txt', 'w') as f:
        for key, value in config.items():
            f.write(f'{key} = {value}\n')
        