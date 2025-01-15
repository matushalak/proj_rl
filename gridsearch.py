# from itertools import product
# from numpy import arange, full, argmax
# from main import main

# # 7 values between 0-1, step 0.25
# vals = full((7, 6), arange(1.2, step = 0.2))

# hyperparams = [list(range(20, 40, 5)) # moving average window size
#                ]+ (vals.T * [1, # monthly add
#                             -1, # PEAK week (sell)
#                             -1, # PEAK weekend (sell)
#                             -1, # off peak week sell
#                             -1, # off peak weekend sell
#                             1, # off peak week buy
#                             1] # off peak weekend buy
#                             ).T.tolist() 

# print(hyperparams)
# param_configs = []
# fitnesses = []
# for i, p in enumerate(product(*hyperparams)):
#     param_configs.append(p)
#     fit = main(path_to_dataset='train.xlsx', PRINT=False, agent_params=p)
#     fitnesses.append(fit)

#     print(p, fit)

# print('Best fitness', max(fitnesses))
# print('Best params', param_configs[argmax(fitnesses)])

from itertools import product
from numpy import arange, full, argmax
from joblib import Parallel, delayed
from main import main

# 7 values between 0-1, step 0.25
vals = full((7, 5), arange(1.25, step=0.25))

hyperparams = [[25, 49, 73]  # moving average window size
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

# Find the best configuration
best_index = argmax(fitnesses)
print('Best fitness', fitnesses[best_index])
print('Best params', param_configs[best_index])