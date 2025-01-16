from agents import HourAgent, Average, AverageHour, RandomAgent
import seaborn as sns
from env import DataCenterEnv
import numpy as np
from utils import preprocess_state
from main import main
from matplotlib import pyplot as plt
from pandas import DataFrame

files = ['train.xlsx', 'validate.xlsx']
names = {0:'Random Agent', 1: 'Hour Agent', 2: 'Moving Average Agent', 3: 'Average-Hour-Weekend-Month Agent'}
results = DataFrame({'Agent':['Random Agent','Hour Agent', 'Moving Average Agent', 'Average-Hour-Weekend-Month Agent'] * 2,
                     'Dataset': ['train']*4 + ['validate']*4,
                     'Yearly Cost': [0]*8})
count = -1
for file in files:
    dataset = file.split('.')[0]
    for ia, Agnt in enumerate([RandomAgent, HourAgent, Average, AverageHour]):
        count += 1
        res, acts = main(path_to_dataset=file, retACTIONS=True, Agent= Agnt)

        results.iloc[count,-1] = -res
        # if count == 2:
        #     breakpoint()
        data = DataCenterEnv(file).test_data
        data = data.melt(id_vars='PRICES', var_name='Hour', value_name='Price')
        data.rename(columns={'PRICES':'Date'}, inplace=True)
        data = data.sort_values(by=['Date', 'Hour']).reset_index(drop=True)
        data['Action'] = acts + [0]
        data['index'] = data.index

        plt.figure()
        plot = sns.scatterplot(data = data, x = 'index', y = 'Price', hue = 'Action', palette = 'coolwarm', legend = False)
        plt.title(f'{names[ia]} {dataset} dataset')
        plt.yscale('log')
        plt.legend()
        norm = plt.Normalize(vmin=data['Action'].min(), vmax=data['Action'].max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
        sm.set_array([])  # This is required for ScalarMappable to work properly
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Actions (MWh)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{names[ia]} {dataset}.png', dpi = 200)
        plt.show()

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
plt.show()