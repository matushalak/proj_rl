from env import DataCenterEnv
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from cma import CMAEvolutionStrategy
from scipy.special import softmax





class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input = None
        self.weights_hidden_input = None
        self.biases_input = None

        self.weights_forget = None
        self.weights_hidden_forget = None
        self.biases_forget = None

        self.weights_cell = None
        self.weights_hidden_cell = None
        self.biases_cell = None

        self.weights_output = None
        self.weights_hidden_output = None
        self.biases_output = None

        self.weights_final = None
        self.biases_final = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def set_params(self, params):
        idx = 0

        # Input Gate Weights (U and W) and Bias
        self.weights_input = params[idx:idx + self.input_size * self.hidden_size].reshape(self.input_size,
                                                                                          self.hidden_size)
        idx += self.input_size * self.hidden_size
        self.weights_hidden_input = params[idx:idx + self.hidden_size * self.hidden_size].reshape(self.hidden_size,
                                                                                                  self.hidden_size)
        idx += self.hidden_size * self.hidden_size
        self.biases_input = params[idx:idx + self.hidden_size]
        idx += self.hidden_size

        # Forget Gate Weights (U and W) and Bias
        self.weights_forget = params[idx:idx + self.input_size * self.hidden_size].reshape(self.input_size,
                                                                                           self.hidden_size)
        idx += self.input_size * self.hidden_size
        self.weights_hidden_forget = params[idx:idx + self.hidden_size * self.hidden_size].reshape(self.hidden_size,
                                                                                                   self.hidden_size)
        idx += self.hidden_size * self.hidden_size
        self.biases_forget = params[idx:idx + self.hidden_size]
        idx += self.hidden_size

        # Cell Gate Weights (U and W) and Bias
        self.weights_cell = params[idx:idx + self.input_size * self.hidden_size].reshape(self.input_size,
                                                                                         self.hidden_size)
        idx += self.input_size * self.hidden_size
        self.weights_hidden_cell = params[idx:idx + self.hidden_size * self.hidden_size].reshape(self.hidden_size,
                                                                                                 self.hidden_size)
        idx += self.hidden_size * self.hidden_size
        self.biases_cell = params[idx:idx + self.hidden_size]
        idx += self.hidden_size

        # Output Gate Weights (U and W) and Bias
        self.weights_output = params[idx:idx + self.input_size * self.hidden_size].reshape(self.input_size,
                                                                                           self.hidden_size)
        idx += self.input_size * self.hidden_size
        self.weights_hidden_output = params[idx:idx + self.hidden_size * self.hidden_size].reshape(self.hidden_size,
                                                                                                   self.hidden_size)
        idx += self.hidden_size * self.hidden_size
        self.biases_output = params[idx:idx + self.hidden_size]
        idx += self.hidden_size

        # Final Output Layer Weights and Bias
        self.weights_final = params[idx:idx + self.hidden_size * self.output_size].reshape(self.hidden_size,
                                                                                           self.output_size)
        idx += self.hidden_size * self.output_size
        self.biases_final = params[idx:idx + self.output_size]


    def forward(self, obs, hidden_state, cell_state):
        #input gate
        i_g = np.dot(obs, self.weights_input) + np.dot(hidden_state, self.weights_hidden_input) + self.biases_input
        i_g = 1 / (1 + np.exp(-np.clip(i_g, -709, 709))) #sigmoid act

        #forget gate
        f_g = np.dot(obs, self.weights_forget) + np.dot(hidden_state, self.weights_hidden_forget) + self.biases_forget
        f_g = 1 / (1 + np.exp(-np.clip(f_g, -709, 709))) #sigmoid act

        #update cell gate
        c_g = np.dot(obs, self.weights_cell) + np.dot(hidden_state, self.weights_hidden_cell) + self.biases_cell
        c_t = np.tanh(c_g)
        cell_state = f_g * cell_state + i_g * c_t #candidate cell state update

        #output gate
        o = np.dot(obs, self.weights_output) + np.dot(hidden_state, self.weights_hidden_output) + self.biases_output
        o = 1 / (1 + np.exp(-np.clip(o, -709, 709))) #sig act

        #update hidden
        hidden_state = o * np.tanh(cell_state)

        #final output
        f_o = np.dot(hidden_state, self.weights_final) + self.biases_final
        return np.tanh(f_o), hidden_state, cell_state




def evaluate(params, input_size, hidden_size, output_size):

    model = LSTM(input_size, hidden_size, output_size)
    model.set_params(params)
    environment = DataCenterEnv("train.xlsx")
    aggregate_reward = 0
    terminated = False
    state = environment.observation()
    timestamps = environment.timestamps
    procesed_state = preprocess_state(state, timestamps)

    hidden_state = np.zeros((1, model.hidden_size))  # Or batch_size x hidden_size
    cell_state = np.zeros((1, model.hidden_size))

    while not terminated:

        action, hidden_state, cell_state = model.forward(procesed_state, hidden_state, cell_state)
        next_state, reward, terminated = environment.step(action)
        next_state = preprocess_state(next_state, timestamps)

        procesed_state = next_state
        aggregate_reward += reward


    return -aggregate_reward


input_size = 8
hidden_size = 10
output_size = 1
param_size = (4 * (input_size * hidden_size)) + \
             (4 * (hidden_size * hidden_size)) + \
             (4 * hidden_size) + \
             (hidden_size * output_size) + \
             output_size
options_model_2 = {'verb_filenameprefix': 'modelLSTM_2_outcmaes_'}
#param_size = np.loadtxt("best_params.txt", delimiter=",")
es = CMAEvolutionStrategy(param_size, 0.5, options_model_2)

counter = 0
while not es.stop():
    solutions = es.ask()
    rewards = [evaluate(params, input_size, hidden_size, output_size) for params in solutions]
    es.tell(solutions, rewards)
    es.logger.add()
    es.disp()

    if counter % 20 == 0:
        best_params = es.result.xbest
        np.savetxt("best_params2.txt", best_params, delimiter=",")
    counter += 1
