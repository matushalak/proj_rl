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


class DQN(nn.Module):
    def __init__(self, state_size, action_size, lr):
        super(DQN, self).__init__()

        self.dense1 = nn.Linear(4, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, action_size)

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
        self.reward_buffer = deque([-200.0], maxlen = 100)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        state = self.env.observation()
        #timestamps = self.env.timestamps
        #state = preprocess_state(state, timestamps)

        for _ in range(self.min_replay_size):
            action = discretize_actions(np.random.uniform(-1, 1))
            next_state, reward, terminated = env.step(action)
            transition = (state, action, reward, terminated, next_state)
            self.replay_buffer.append(transition)
            state = next_state

            if terminated:
                terminated = False
                state = env.observation()
                #timestamps = env.timestamps
                #state = preprocess_state(state, timestamps)
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
        self.online_net = DQN(8, 1, lr).to(self.device)

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

        loss = F.smooth_l1_loss(action_q_values, target.detach())

        self.online_net.optimizer.zero_grad()
        loss.backward()
        self.online_net.optimizer.step()



discount_rate = 0.99
batch_size = 32
buffer_size = 50000
min_replay_size = 1000
epsilon_start = 1
epsilon_end = 0.05
epsilon_decay = 10000
max_episodes = 250000

lr = 5e-4
env  = DataCenterEnv("train.xlsx")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dqn_agent = DQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size)



def training(env, agent, max_episodes, target_ = False, seed = 42):
    aggregate_reward = 0
    avg_reward = []
    terminated = False
    state = env.observation()
    #timestamps = env.timestamps
    #state = preprocess_state(state, timestamps)

    for step in range(max_episodes):



        action, epsilon = agent.choose_action(step, state)



        next_state, reward, terminated = env.step(action)
        trainsition = (state, action, reward, terminated, next_state)
        agent.replay_memory.add_data(trainsition)
        state = next_state
        aggregate_reward += reward
        print(aggregate_reward)
        if terminated:
            state = env.observation()
            #timestamps = env.timestamps
            #state = preprocess_state(state, timestamps)
            env = env
            print(aggregate_reward)
            agent.replay_memory.add_reward(aggregate_reward)
            aggregate_reward = 0

        agent.learn(batch_size)

        if (step+1) % 100 == 0:
            avg_reward.append(np.mean(agent.replay_memory.reward_buffer))


        if target_:
            target_update_frequency = 250
            if step % target_update_frequency == 0:
                dagent.update_target_net()

        if (step+1) % 1000 == 0:
            print(20 * '--')
            print('Step', step)
            print('Epsilon', epsilon)
            print('Avg Rew', np.mean(agent.replay_memory.reward_buffer))
            print(avg_reward)
            print()

    return avg_reward





avg_rew_dqn = training(env, dqn_agent, max_episodes)














