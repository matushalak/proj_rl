from collections import deque

from env import DataCenterEnv
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *
from cma import CMAEvolutionStrategy
from scipy.special import softmax
import random
import matplotlib.pyplot as plt

seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
td_error = []
td = 0

class DQN(nn.Module):
    def __init__(self, state_size, action_size, lr):
        super(DQN, self).__init__()

        self.dense1 = nn.Linear(8, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, 5)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):


        x = torch.tanh(self.dense1(state))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = self.dense4(x)

        return x

class experience_replay:
    def __init__(self, env, buffer_size, min_replay_size = 1000, seed=132):
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([-7214030.749999992], maxlen = 100)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        state = self.env.observation()
        timestamps = self.env.timestamps
        state = preprocess_state(state, timestamps)

        for _ in range(self.min_replay_size):
            action = discretize_actions(np.random.uniform(-1, 1))
            next_state, reward, terminated = env.step(action)
            next_state = preprocess_state(next_state, timestamps)
            transition = (state, action, reward, terminated, next_state)
            self.replay_buffer.append(transition)
            state = next_state

            if terminated:
                terminated = False
                state = env.observation()
                timestamps = env.timestamps
                state = preprocess_state(state, timestamps)
                env = env

        print('init with random transitions done!')

    def add_data(self, data):
        self.replay_buffer.append(data)

    def sample(self, batch_size):

        transitions = random.sample(self.replay_buffer, batch_size)

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

        return states_t, actions_t, rewards_t, dones_t, next_states_t

    def add_reward(self, reward):
        self.reward_buffer.append(reward)


class DQNAgent:
    def __init__(self, env, device, epsilon_decay,
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed=132):
        self.env = env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.lr = lr
        self.buffer_size = buffer_size

        self.replay_memory = experience_replay(self.env, self.buffer_size, seed=seed)
        self.online_net = DQN(4, 5, lr).to(self.device)

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

    def learn(self, batch_size):

        states, actions, rewards, dones, next_states = self.replay_memory.sample(batch_size)

        target_q_values = self.online_net(next_states)
        max_target_q_val = target_q_values.max(dim=1, keepdim=True)[0]

        target = rewards + self.discount_rate * max_target_q_val * (1 - dones)

        #loss
        q_values = self.online_net(states)
        #print(actions)
        action_q_values = torch.gather(q_values, 1, actions.long())
        #print(f"Q-values: {q_values}")
        #print(f"Selected Q-values: {action_q_values}")
        td_error.append((target - action_q_values).abs().mean().item())
        print(f"TD Error: {(target - action_q_values).abs().mean().item()}")
        loss = F.smooth_l1_loss(action_q_values, target.detach())

        self.online_net.optimizer.zero_grad()
        loss.backward()
        self.online_net.optimizer.step()
        #print(f"Updated Q-values: {q_values}")

class DDQNAgent:
    def __init__(self, env, device, epsilon_decay,
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed=132):
        self.env = env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.lr = lr
        self.buffer_size = buffer_size

        self.replay_memory = experience_replay(self.env, self.buffer_size, seed=seed)
        self.online_net = DQN(4, 5, lr).to(self.device)

        self.target_net = DQN(4, 5, lr).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

    def choose_action(self, step, state, greedy=False):
        if step % 25000 == 0:
            epsilon = epsilon_start
        else:
            epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        random_sample = random.random()

        if (random_sample <= epsilon) and not greedy:
            # Random action
            action = discretize_actions(np.random.uniform(-1, 1))

        else:
            state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            q_vals = self.online_net(state.unsqueeze(0))

            max_q_index = torch.argmax(q_vals, dim=1)[0]
            action = max_q_index.detach().item()

        return action, epsilon

    def return_q_values(self, state):

        obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        q_values = self.online_net(obs_t.unsqueeze(0))

        return torch.max(q_values).item()

    def learn(self, batch_size):

        states, actions, rewards, dones, next_states = self.replay_memory.sample(batch_size)

        next_action = torch.argmax(self.online_net(next_states), dim=1)

        #target_q_values = self.target_net(next_states)
        #max_target_q_val = target_q_values.max(dim=1, keepdim=True)[0]

        max_target_q_val = self.target_net(next_states).gather(1, next_action.unsqueeze(-1))

        target = rewards + self.discount_rate * max_target_q_val * (1 - dones)

        # loss
        q_values = self.online_net(states)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions.long())


        #td_error.append((target - action_q_values).abs().mean().item())
        #td = (target - action_q_values).abs().mean().item()

        loss = F.smooth_l1_loss(action_q_values, target.detach())

        self.online_net.optimizer.zero_grad()
        loss.backward()
        self.online_net.optimizer.step()
        # print(f"Updated Q-values: {q_values}")
        #for param in self.online_net.parameters():
         #   if param.grad is not None:
          #      print(f"Gradient Norm: {param.grad.norm().item()}")

        td_error.append((target - action_q_values).abs().mean().item())


    def update_target(self):

        self.target_net.load_state_dict(self.online_net.state_dict())



def calculate_rewards(storage_change, storage_level, unmet_demand, excess_energy, max_storage):
    reward = 0
    if storage_change > 0:
        reward += storage_change

    if storage_change < 0:
        reward -= abs(storage_change)



    reward -= unmet_demand * 5

    if excess_energy > 50:
        reward -= (excess_energy - 50) * 2


    return reward


discount_rate = 0.95
0.99
#vizualize q val
#reward shaping
#start with discount factor of 0 to make sure
batch_size = 32
buffer_size = 500000
min_replay_size = 1000
epsilon_start = 1
epsilon_end = 0.1
epsilon_decay = 1000000
max_episodes = 2000000

#lr = 5e-4
lr = 0.0005
env  = DataCenterEnv("train.xlsx")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dqn_agent = DQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size)
dagent = DDQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size)

rew = []

class reward_scaler:
    def __init__(self):
        self.min_reward = float("inf")
        self.max_reward = float("-inf")

    def update(self, reward):
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)

    def scale_reward(self, reward):
        if self.max_reward > self.min_reward:
            return (reward - self.min_reward) / (self.max_reward - self.min_reward)  # Scale to [0, 1]
        else:
            return 0


def training(env, agent, max_episodes, target_ = True, seed = 42):
    aggregate_reward = 0
    avg_reward = []
    terminated = False
    state = env.observation()
    print(state)
    timestamps = env.timestamps
    state = preprocess_state(state, timestamps)
    storage_level = 0
    max_storage = 110
    excess_buffer = 50
    daily_requirement = 120
    total_energy_bought = 0
    max_price = 2500
    min_price = 0.01


    state[0] = state[0]/max_storage
    state[1] = (state[1] - min_price) / (max_price - min_price)
    state[2] = state[2] / 24
    state[3] = state[3] / 1096
    state[4] = state[4] / 31
    state[5] = state[5] / 6
    state[6] = state[6] / 53
    state[7] = state[7] / 12
    print(state)
    rew_scaler_true = reward_scaler()
    rew_scaler_shaped = reward_scaler()

    for step in range(max_episodes):



        action, epsilon = agent.choose_action(step, state)


        next_state, reward, terminated = env.step(normalize_actions(action))

        next_state = preprocess_state(next_state, timestamps)
        next_state[0] = next_state[0] / max_storage
        next_state[1] = (next_state[1] - min_price) / (max_price - min_price)
        next_state[2] = next_state[2] / 24
        next_state[3] = next_state[3] / 1096
        next_state[4] = next_state[4] / 31
        next_state[5] = next_state[5] / 6
        next_state[6] = next_state[6] / 53
        next_state[7] = next_state[7] / 12


        storage_level = state[0]
        next_storage_level = next_state[0]
        storage_change = next_storage_level - storage_level

        price = state[1]
        hour = state[2]

        total_energy_bought = max(0, daily_requirement - storage_level)
        unmet_demand = max(0, daily_requirement - total_energy_bought)

        excess_energy = max(0, next_storage_level - excess_buffer)

        shaped_reward = calculate_rewards(storage_change, storage_level, unmet_demand, excess_energy, max_storage)

        rew_scaler_true.update(reward)
        rew_scaler_shaped.update(shaped_reward)

        scaled_rew_true = rew_scaler_true.scale_reward(reward)
        scaled_rew_shaped = rew_scaler_shaped.scale_reward(shaped_reward)


        final_reward = 0.3 * scaled_rew_shaped + (1 - 0.3) * scaled_rew_true
        #print(scaled_rew_true, scaled_rew_shaped, final_reward)
        trainsition = (state, action, final_reward, terminated, next_state)
        agent.replay_memory.add_data(trainsition)
        rew.append(reward)
        state = next_state

        #print(reward)
        #print(shaped_reward)
        #print(final_reward)
        #print('--'*20)
        aggregate_reward += reward

        if terminated:
            #print(f"Environment terminated at day {env.day}, hour {env.hour}")
            env = DataCenterEnv("train.xlsx")
            state = env.observation()

            timestamps = env.timestamps
            state = preprocess_state(state, timestamps)
            env = env
            print(f"Episode Reward: {aggregate_reward}")
            print(f"Steps: {step}")
            print(f"Td Error: {td_error[-1]}")
            agent.replay_memory.add_reward(aggregate_reward)
            aggregate_reward = 0
            epsilon = epsilon_start



        agent.learn(batch_size)

        if (step+1) % 100 == 0:
            avg_reward.append(np.mean(agent.replay_memory.reward_buffer))


        if target_:
            target_update_frequency = 250
            if step % target_update_frequency == 0:
                dagent.update_target()



        #if (step+1) % 1000 == 0:
        #    print(20 * '--')
         #   print('Step', step)
          #  print('Epsilon', epsilon)
          #  print('Avg Rew', np.mean(agent.replay_memory.reward_buffer))
           # #print(avg_reward)
            #print(td_error[-1])
            #print()

    return avg_reward


#make sure we have a good tabuar rl
#make sure agent has visited all discrete states
#get reward shaping right
#--> something that makes sense --> validate whether it makes sense by test on validate
#add internal reward when storage level increase and similarly lower reward when storage level decreases


#avg_rew_dqn = training(env, dqn_agent, max_episodes)
average_rewards_ddqn = training(env, dagent, max_episodes, target_ = True)


plt.plot(td_error)
plt.xlabel('Training Steps')
plt.ylabel('TD Error')
plt.title('Temporal Difference Error Over Training')
plt.show()












