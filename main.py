from env import DataCenterEnv
import numpy as np
import argparse
from utils import preprocess_state, extract_bf_number
from agents import RandomAgent, AverageHour, QAgent
import os
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns


def main(path_to_dataset:str, retrain:bool = False, PRINT:bool = False, agent_params:list|bool = False, retACTIONS: bool = False, 
         Agent:object = QAgent) -> float:
    # 1) Prepare / train agent
    # hardcoded agent by hour
    if agent_params:
        agent = AverageHour(*agent_params)

    # QAgent <<- OUR RL agent 
    elif Agent == QAgent:
        Qtables = [file for file in os.listdir() if file.startswith('Qtable')]
        if len(Qtables) == 0 or retrain == True:
            agent = Agent()
            QTABLE = agent.train(dataset = 'train.xlsx')
        
        else:
            # Find the file with the highest BF (best fitness) number
            highest_bf_file = max(Qtables, key=extract_bf_number)
            print(f'Using this Qtable: {highest_bf_file}')
            agent = Agent(Qtable_dir = highest_bf_file)
            # breakpoint()
    
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
        return aggregate_reward / nyears, actions, agent
    else:
        return aggregate_reward / nyears

# had to put here because in utils.py had issue with circular imports 
def performance_plots(files:list[str]):
    # files = ['train.xlsx', 'validate.xlsx']
    names = {0:'Random Agent', 1: 'Average-Hour-Weekend-Month Agent', 2:'Q-learning Agent'}
    results = DataFrame({'Agent':['Random Agent','Average-Hour-Weekend-Month Agent', 'Q-learning agent'] * 2,
                        'Dataset': ['train']*3 + ['validate']*3,
                        'Yearly Cost': [0]*6})
    count = -1
    for file in files:
        dataset = file.split('.')[0]
        for ia, Agnt in enumerate([RandomAgent, AverageHour, QAgent]):
            count += 1
            
            if Agnt is QAgent:
                res, acts, agent = main(path_to_dataset=file, retACTIONS=True, Agent= Agnt)
            else:
                res, acts, _ = main(path_to_dataset=file, retACTIONS=True, Agent= Agnt)

            results.iloc[count,-1] = -res
            # if count == 2:
            data = DataCenterEnv(file).test_data
            data = data.melt(id_vars='PRICES', var_name='Hour', value_name='Price')
            data.rename(columns={'PRICES':'Date'}, inplace=True)
            data = data.sort_values(by=['Date', 'Hour']).reset_index(drop=True)
            data['Action'] = acts + [0]
            data['index'] = data.index
            
            ############################################
            # Q table plot (adapted from tutorial notebook)
            if Agnt is QAgent and file == 'train.xlsx':
                # Qt_HIprice, Qt_LOprice, Qt_Mprice = agent.Qtable[:,:,0,:], agent.Qtable[:,:,1,:], agent.Qtable[:,:,2,:]
                
                # loop through price levels and 1 plot / price level
                for price_cat in range(agent.Qtable.shape[2]):
                    prices = {0:"Above",
                              1:"Below",
                              2:"Around"}
                    
                    X, Y = np.meshgrid(np.arange(agent.hours.size + 1), np.arange(agent.storage.size + 1))
                    # breakpoint()
                    Z = np.zeros((len(X), len(Y)))
                    A = np.zeros(Z.shape)

                    # hour
                    for h in range(len(X)):
                        # storage
                        for s in range(len(Y)):
                            a = np.argmax(agent.Qtable[h, s, price_cat,:])
                            A[h][s] = agent.actions[a] # store "best"actions for hour, storage, price pair
                            Z[h][s] = agent.Qtable[h, s, price_cat, a] # store corresponding Qvalues

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    # Normalize action values for colormap mapping
                    norm = plt.Normalize(vmin=-1, vmax=1)  # Ensure -1 is blue, 0 is white, 1 is red
                    cmap = plt.cm.coolwarm  # Define colormap

                    # Map action values to colors
                    colors = cmap(norm(A))

                    # Create surface plot with manually mapped colors
                    surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0, antialiased=False)
                    ax.view_init(elev=38, azim=-126) 
                    
                    # Remove default ticks and set custom labels
                    ax.set_xticks([0, 1, 2, 3, 4])  # Positions
                    ax.set_xticklabels(["Morning", "Noon", "Afternoon", "Evening", "Night"], fontsize=10)

                    ax.set_yticks([0, 1, 2, 3, 4])  # Positions
                    ax.set_yticklabels(["< 40", "< 80", "< 120", "< 150", "> 150"], fontsize=10)

                    # Optionally remove the tick marks
                    ax.tick_params(axis='x', which='both', bottom=False, top=False)  # Remove x-axis ticks
                    ax.tick_params(axis='y', which='both', left=False, right=False)  # Remove y-axis ticks
                    
                    # Set labels
                    ax.set_xlabel('Hour', fontsize=12)
                    ax.set_ylabel('Storage Level', fontsize=12)
                    ax.set_zlabel('Q-Value', fontsize=12)
                    ax.set_title(f'Q-Table (Price Category: {prices[price_cat]} moving weekly average)', fontsize=14)

                    # Create a separate colorbar for actions
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])  # Dummy array for colorbar
                    cbar = fig.colorbar(sm, ax = ax, shrink=0.5, aspect=5)
                    cbar.set_label('Best Action ±(10 MWh)', fontsize=12)

                    plt.tight_layout()
                    plt.savefig(f'Qtable{price_cat}.png', dpi = 200)
                    # plt.show()
                    plt.close()
            
            #################
            # Time series plot
            plt.figure()
            # only plot x days
            if Agnt: #is QAgent and file == 'validate.xlsx':
                # breakpoint()
                slice_to_plot = data.iloc[:100,:]#[-24*31:,:]#[100:7 * 24 + 100,:]#[-72:,:]#[-1096:-1000,:]
            
                plot = sns.scatterplot(data = slice_to_plot, x = 'index', y = 'Price', hue = 'Action', hue_norm=(-1,1), palette = 'coolwarm',edgecolor='k', legend = False)
                plt.title(f'{names[ia]} {dataset} dataset')
                # plt.yscale('log')
                plt.legend()
                norm = plt.Normalize(vmin=-1, vmax=1)
                sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
                sm.set_array([])  # This is required for ScalarMappable to work properly
                
                cbar = plt.colorbar(sm, ax=plt.gca())
                cbar.set_label('Actions (MWh)', fontsize=12)
                
                plt.tight_layout()
                plt.savefig(f'{names[ia]} {dataset}_{slice_to_plot.shape[0]//24}DAYS.png', dpi = 200)
                # plt.show()
                plt.close()

    ###################################
    # Overall AGENTS comparison barplot
    plt.figure(figsize=(10,10))
    bp = sns.barplot(data = results, x = 'Dataset', y = 'Yearly Cost', hue = 'Agent', legend=True)
    # Add data labels
    for bar in bp.patches:
        bp.annotate(
            format(bar.get_height(), '.2e'),  # Format the value
            (bar.get_x() + bar.get_width() / 2., bar.get_height()),  # Position the label
            ha='center', va='bottom', size=9  # Alignment and font size
        )
    plt.title('Yearly Costs across datasets & agents')
    plt.legend(loc = 'lower right', facecolor = 'white', framealpha = .95)
    bp.set_ylabel('Yearly Cost (Millions $)')
    plt.tight_layout()
    plt.savefig(f'RESULTS_train-vs-validate.png', dpi = 200)
    # plt.show()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='validate.xlsx')
    args.add_argument('--retrain', type=bool, default=False)
    args.add_argument('--plots', type=bool, default=False)
    args = args.parse_args()

    np.set_printoptions(suppress=True, precision=2)
    path_to_dataset = args.path
    reTRAIN = args.retrain

    
    main(path_to_dataset=path_to_dataset, retrain=reTRAIN)
    
    # make & save plots
    if args.plots:
        if path_to_dataset != 'validate.xlsx':
            performance_plots(files =['train.xlsx', path_to_dataset])
        else:
            performance_plots( files = ['train.xlsx', 'validate.xlsx'])