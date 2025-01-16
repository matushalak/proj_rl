from betterenv import DataCenterEnv
# from env import DataCenterEnv
import pandas as pd
from numpy import arange
import seaborn as sns
import matplotlib.pyplot as plt

# Looking at features
# Hour, Total Day, Calendar Day, Weekday, Week, Month
# maybe look at variance (in price) explained by each of those "predictors"

Validate = DataCenterEnv('validate.xlsx')
Train = DataCenterEnv('train.xlsx')

# Total = pd.concat([Train.test_data, Validate.test_data])
# Total = Total.melt(id_vars='PRICES', var_name='Hour', value_name='Price')
# Total.rename(columns={'PRICES':'Date'}, inplace=True)

# sns.histplot(data = Total, x = 'Price', bins = 100, hue='', log_scale=True, stat='percent')
# # plt.xlim(5,500)
# plt.tight_layout()
# plt.show()

DateInfo = Train.timestamps

melted_data = Train.test_data.melt(id_vars='PRICES',
                                var_name= 'Hour',
                                value_name='Price')
melted_data.rename(columns={'PRICES':'Date'}, inplace=True)

melted_data['Hour'] = melted_data['Hour'].str.replace('Hour ', '').astype(int)

features_df = pd.DataFrame({'Date':DateInfo,
                            'Total Day': arange(DateInfo.size), # 0 - 1095, 3 years of data
                            'Calendar Day':DateInfo.apply(lambda x: x.day), # 1 - 31
                            'Weekday':DateInfo.apply(lambda x: x.weekday()), # Mon {0} - Sun {6}
                            'Week':DateInfo.apply(lambda x: x.week), # 1 - 53
                            'Month':DateInfo.apply(lambda x: x.month)}) # Jan {1} - Dec {12}

Features = pd.merge(melted_data, features_df, on = 'Date', how = 'left')
# More logical ordering
reorder =['Date', 'Hour', 'Calendar Day', 'Weekday', 'Week', 'Month', 'Total Day', 'Price']
Features = Features[reorder]

DateInfo2 = Validate.timestamps

melted_data2 = Validate.test_data.melt(id_vars='PRICES',
                                var_name= 'Hour',
                                value_name='Price')
melted_data2.rename(columns={'PRICES':'Date'}, inplace=True)

melted_data2['Hour'] = melted_data2['Hour'].str.replace('Hour ', '').astype(int)

features_df2 = pd.DataFrame({'Date':DateInfo2,
                            'Total Day': arange(DateInfo2.size), # 0 - 1095, 3 years of data
                            'Calendar Day':DateInfo2.apply(lambda x: x.day), # 1 - 31
                            'Weekday':DateInfo2.apply(lambda x: x.weekday()), # Mon {0} - Sun {6}
                            'Week':DateInfo2.apply(lambda x: x.week), # 1 - 53
                            'Month':DateInfo2.apply(lambda x: x.month)}) # Jan {1} - Dec {12}

Features2 = pd.merge(melted_data2, features_df2, on = 'Date', how = 'left')
# More logical ordering
reorder =['Date', 'Hour', 'Calendar Day', 'Weekday', 'Week', 'Month', 'Total Day', 'Price']
Features2 = Features2[reorder]
# Plot One Feature
# one_plot = sns.lineplot(data = Features, x = 'Month', y = 'Price', marker = 'o')
# one_plot.set_xticks(arange(1, 13))
# plt.show()

colnames = Features.columns
fig, axes = plt.subplots(3, 3)
fig.set_size_inches(10,10)
row, col = 0, 0
for i, predictor in enumerate(colnames[1:-1]):
    ax = axes[row][col]
    sns.lineplot(data = Features, x = predictor, y = 'Price', ax = ax, marker = 'o')
    sns.lineplot(data = Features2, x = predictor, y = 'Price', ax = ax, marker = '^')
    row += 1

    if row > 2 :
        col += 1
        row = col
plt.tight_layout()
plt.savefig('FEATURES.png', dpi = 200)
plt.show()
