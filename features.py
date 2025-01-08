from env import DataCenterEnv
import pandas as pd
from numpy import arange
import seaborn as sns
import matplotlib.pyplot as plt

# Looking at features
# Hour, Total Day, Calendar Day, Weekday, Week, Month
# maybe look at variance (in price) explained by each of those "predictors"

Train = DataCenterEnv('train.xlsx')
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
reorder =['Date', 'Hour', 'Total Day', 'Calendar Day', 'Weekday', 'Week', 'Month', 'Price']
Features = Features[reorder]

byHour = Features.groupby('Hour')
breakpoint()
sns.scatterplot(byHour, x = 'Hour', y = 'Price')
plt.show()