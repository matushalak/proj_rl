from itertools import product
from numpy import arange, full, argmax
from pandas import DataFrame
from joblib import Parallel, delayed
from main import main

# 7 values between 0-1, step 0.25
vals = full((7, 5), arange(1.25, step=0.25))

hyperparams = [[25, 49, 73, 97, 121],  # moving average window size
               ] + (vals.T * [1,  # monthly add
                              -1,  # PEAK week (sell)
                              -1,  # PEAK weekend (sell)
                              -1,  # off peak week sell
                              -1,  # off peak weekend sell
                              1,  # off peak week buy
                              1]  # off peak weekend buy
                              ).T.tolist()

print(hyperparams)
param_configs = list(product(*hyperparams))


# Function to evaluate fitness for a single configuration
def evaluate_fitness(config):
    fit = main(path_to_dataset='train.xlsx', PRINT=False, agent_params=config)
    print(config, fit)
    return config, fit


# Parallel computation
results = Parallel(n_jobs=-1)(delayed(evaluate_fitness)(p) for p in param_configs)

# Extract configurations and fitnesses
param_configs, fitnesses = zip(*results)

results = DataFrame({'params': param_configs,
                     'cost_per_year': fitnesses})

results.to_csv('gridsearch_results.csv')

# Find the best configuration
best_index = argmax(fitnesses)
print('Best fitness', fitnesses[best_index])
print('Best params', param_configs[best_index])