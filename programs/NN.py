
from env import DataCenterEnv
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from cma import CMAEvolutionStrategy
from scipy.special import softmax





class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None
        self.weights3 = None
        self.bias3 = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def set_params(self, params):
        # Weights1
        self.weights1 = params[:self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
        # Bias1
        self.bias1 = params[self.input_size * self.hidden_size:self.input_size * self.hidden_size + self.hidden_size]

        # Weights2
        start_w2 = self.input_size * self.hidden_size + self.hidden_size
        end_w2 = start_w2 + self.hidden_size * self.hidden_size

        self.weights2 = params[start_w2:end_w2].reshape(self.hidden_size, self.hidden_size)
        # Bias2
        start_b2 = end_w2
        end_b2 = start_b2 + self.hidden_size
        self.bias2 = params[start_b2:end_b2]

        # Weights3
        start_w3 = end_b2
        end_w3 = start_w3 + self.hidden_size * self.output_size
        self.weights3 = params[start_w3:end_w3].reshape(self.hidden_size, self.output_size)
        # Bias3
        self.bias3 = params[end_w3:end_w3 + self.output_size]

    def forward(self, obs):
        h1 = np.dot(obs, self.weights1) + self.bias1
        h1 = np.tanh(h1)
        h2 = np.dot(h1, self.weights2) + self.bias2
        h2 = np.tanh(h2)

        output = np.dot(h2, self.weights3) + self.bias3
        return np.tanh(output)




def evaluate(params, input_size, hidden_size, output_size):

    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.set_params(params)
    environment = DataCenterEnv("train.xlsx")
    aggregate_reward = 0
    terminated = False
    state = environment.observation()
    timestamps = environment.timestamps
    procesed_state = preprocess_state(state, timestamps)


    while not terminated:

        action = model.forward(procesed_state)
        next_state, reward, terminated = environment.step(action)
        next_state = preprocess_state(next_state, timestamps)

        procesed_state = next_state
        aggregate_reward += reward


    return -aggregate_reward


input_size = 8
hidden_size = 10
output_size = 1
param_size = (input_size * hidden_size) + hidden_size + (hidden_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size
opt_model_2 = {'verb_filenameprefix': 'modelNN_2_outcmaes_'}

param_size = np.loadtxt("best_params_NN.txt", delimiter=",")


es = CMAEvolutionStrategy(param_size, 0.5, opt_model_2)

counter = 0
while not es.stop():
    solutions = es.ask()
    rewards = [evaluate(params, input_size, hidden_size, output_size) for params in solutions]
    es.tell(solutions, rewards)
    es.logger.add()
    es.disp()

    if counter % 20 == 0:
        best_params = es.result.xbest
        np.savetxt("best_params_NN_2.txt", best_params, delimiter=",")
        counter += 1



