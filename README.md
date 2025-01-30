# proj_rl
This repository contains the code for Reinforcement Learning Project MSc. course at VU (2025).

## Scripts / Files
- `main.py` runs the Tabular RL agent with the best available Q-table present in the repository on the provided data. The behavior of the script can be modified with the command-line arguments below:
    - `python main.py --retrain True` will retrain the agent on the Training dataset (*train.xlsx*) and upon finalization of training will run the agent on the validation dataset (*validate.xlsx*). This process will also produce plots of training progress showing the yearly cost on both the training anda validation datasets. By default this is flag false and just calling `python main.py` will use the Q-table in the repository with the best performance
    - `python main.py --path test_file_name.xlsx`  will employ the QAgent on provided dataset and return the Total, as well as, Yearly costs for the given dataset, provided that *test_file_name.xlsx* is an excel file exactly in the same format as *train.xlsx* and *validate.xlsx*. By default `python main.py` without the --path flag specifying a test file will use the *validate.xlsx* dataset that was provided for the project.
    - `python main.py --plots test_file_name.xlsx` will produce plots (1. Time series plot of the first 4 days, 2. Q Table visualization, 3. Overall Barplot comparing performance of Random, Our Best Heuristic and Our Q-Agent) of the results obtained by a Random Agent / our best Heuristic Agent / our RL Q-learning Agent both on the training dataset and the validation / test dataset if another dataset is provided under the `--path` flag. By default this is false and no plots will be produced.
        - *INTENDED USAGE for RUNNING THE ALGORITHM*:
            - for seeing the results of our Q-Agent and provided Qtable on the test dataset: 
            `python main.py --path test.xlsx --plots True`
            - for seeing the retraining plots and retraining the agent:
            `python main.py --retrain True --path test.xlsx -- plots True`
            - for just quickly looking at the results: `python main.py --path test.xlsx`

- `env.py` the provided environment, we did not modify it

- `agents.py` contrains classes for our heurisitic baseline agents and the Tabular RL Q-learning agent. Our other baseline or RL agents (LSTM, EA NN baselines and DDQN RL agent) are not provided here because they required their own scripts.

- `utils.py` contains two utility functions used in main.py and agents.py

- `requirements.txt` this file should contains all the necessary packages and works on MacOS, but I cannot guarantee how it will behave on windows / linux

## Folders
- *results-PLOTS* contains the plots used in our report
- *training plots-REPLICATED PERFROMANCE* contains training plot examples of 5 independent replications of our Q-learning algorithm achieving performance close to / on the level of our best baseline.

## Tables
- *train.xlsx* contains the training dataset, this file must be present to enable retraining and making of plots which compare performance on provided dataset with training performance
- *validate.xlsx* this is the validation dataset, which is used by default to measure performance during and after training
- *Qtable_300sims_epsd100lrd200_df0.99_BF-1.741.npy* is the Q-table we provide for our agent. If the agent is retrained, a new Q-table will appear with a different number at the end after BF (which indicates average of yearly cost on training and validation datasets).



