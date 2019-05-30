import time
from scipy.optimize import minimize
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import itertools


symbols = ['ARGX.BR', 'BAR.BR', 'GLPG.AS', 'UMI.BR', 'WDP.BR']
weights = [0.21860023, 0.11065033, 0.18923859, 0.35871712, 0.12279372]

indexReference = "^BFX"

start = datetime(2018, 1, 1)
end = datetime(2018, 12, 31)

datasets = []
for symbol in symbols:
        print("Fetching data for {}".format(symbol))
        try:
            f = web.DataReader(symbol, 'yahoo',  start, end)
            f['ticker']=np.full(f['Adj Close'].count(),symbol)
            f=f.drop(["High","Low","Open","Volume","Close"],axis=1)
            print("Fetched {}".format(f.shape))
            if f['Adj Close'].count() >=250:
                datasets.append(f)
        except:
            print("Error:{}".format(symbol))
data = pd.concat(datasets)
table = data.pivot(columns='ticker')




f = web.DataReader(indexReference, 'yahoo',  start, end)
f['ticker']=np.full(f['Adj Close'].count(),indexReference)
indexData=f.drop(["High","Low","Open","Volume","Close"],axis=1)
tableIndex = indexData.pivot(columns='ticker')

#print(tableIndex)
#get the first value of the indexReference at day One
indexRefValue = tableIndex.iloc[0,0]
shares = []

#print(indexRefValue)
shares = [(weights[symbols.index(col[1])]*indexRefValue)/table.iloc[0,table.columns.get_loc(col)] for col in table.columns]
#print("Shares--------")
#print(shares)
#print("---------------")
table['portfolio'] = sum(table[col]*shares[symbols.index(col[1])] for col in table.columns)




plt.style.use('seaborn-dark')
ax = table.plot.line(y='portfolio',c='red', figsize=(10, 8), grid=True)
tableIndex.plot.line(y='Adj Close', c='blue', ax=ax)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Index Comparison')
plt.show()
winst = table['portfolio'][-1]-table['portfolio'][0]
roi = winst / table['portfolio'][0]
print("Winst:{} ROI {}".format(winst, roi))