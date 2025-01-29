from agents import AverageHour, RandomAgent, QAgent
import seaborn as sns
from env import DataCenterEnv
import numpy as np

from main import main
from matplotlib import pyplot as plt
from pandas import DataFrame

def performance_plots():
    files = ['train.xlsx', 'validate.xlsx']
    names = {0:'Random Agent', 1: 'Average-Hour-Weekend-Month Agent', 2:'Q-learning Agent'}
    results = DataFrame({'Agent':['Random Agent','Average-Hour-Weekend-Month Agent', 'Q-learning agent'] * 2,
                        'Dataset': ['train']*3 + ['validate']*3,
                        'Yearly Cost': [0]*6})
    count = -1
    for file in files:
        dataset = file.split('.')[0]
        for ia, Agnt in enumerate([RandomAgent, AverageHour, QAgent]):
            count += 1
            res, acts = main(path_to_dataset=file, retACTIONS=True, Agent= Agnt)

            results.iloc[count,-1] = -res
            # if count == 2:
            data = DataCenterEnv(file).test_data
            data = data.melt(id_vars='PRICES', var_name='Hour', value_name='Price')
            data.rename(columns={'PRICES':'Date'}, inplace=True)
            data = data.sort_values(by=['Date', 'Hour']).reset_index(drop=True)
            data['Action'] = acts + [0]
            data['index'] = data.index
            
            # if Agnt == QAgent and file == 'validate.xlsx':
            #     breakpoint()

            plt.figure()
            # only plot x days
            slice_to_plot = data.iloc[-72:,:]#[100:7 * 24 + 100,:]#[-72:,:]#[-1096:-1000,:]
        
            plot = sns.scatterplot(data = slice_to_plot, x = 'index', y = 'Price', hue = 'Action', hue_norm=(-1,1), 
                                palette = 'coolwarm',edgecolor='k', legend = False)
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
    plt.show()