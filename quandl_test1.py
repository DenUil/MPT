import quandl, os
import pandas as pd

data = None
if not os.path.isfile('MergentDataSet.pkl'):

    quandl.ApiConfig.api_key = '4L9pz5K5udKU-emCMi9f'
    selected = ['CNP', 'F', 'WMT', 'GE', 'TSLA']
    num_assets = len(selected)
    data = quandl.get_table('MER/F1', paginate=True)
    data.to_pickle("MergentDataSet.pkl")

else:
    data = pd.read_pickle("MergentDataSet.pkl")


print(data.loc[data['country']=="BEL"]['ticker'].unique())