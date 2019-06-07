import time
from scipy.optimize import minimize
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import itertools


symbols = ['IRPC.BK', 'KTC.BK', 'ORI.BK', 'RS.BK', 'TFG.BK']
weights = [0.27329778528836024, 0.22622808913118347, 0.3172835831004925, 0.010000000000000009, 0.17319054247996385]

indexReference = "^SET50.BK"

start = datetime(2018, 1, 1)
end = datetime.now()

datasets = []
for symbol in symbols:
        print("Fetching data for {}".format(symbol))
        try:
            f = web.DataReader(symbol, 'yahoo',  start, end)
            f['ticker']=np.full(f['Adj Close'].count(),symbol)
            f=f.drop(["High","Low","Open","Volume","Close"],axis=1)
            print("Fetched {}".format(f.shape))
            if f['Adj Close'].count() >=240:
                datasets.append(f)
        except:
            print("Error:{}".format(symbol))
data = pd.concat(datasets)
table = data.pivot(columns='ticker')




#f = web.DataReader(indexReference, 'yahoo',  datetime(2018, 1, 1), end)
#f['ticker']=np.full(f['Adj Close'].count(),indexReference)
#indexData=f.drop(["High","Low","Open","Volume","Close"],axis=1)
#tableIndex = indexData.pivot(columns='ticker')

f = pd.read_csv('/home/robin/MPT/SET50_Historical_Data_20180101_20190601.csv')
f['ticker']=np.full(f['Price'].count(),indexReference)
indexData=f.drop(["High","Low","Open","Vol.","Change %"],axis=1)
indexData["Date"]= pd.to_datetime(indexData["Date"])
indexData = indexData.set_index('Date')
tableIndex = indexData.pivot(columns='ticker')


print(tableIndex)
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
tableIndex.plot.line(y='Price', c='blue', ax=ax)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Index Comparison')
plt.show()
winst = table['portfolio'][-1]-table['portfolio'][0]
roi = winst / table['portfolio'][0]
print("Winst:{} ROI {}".format(winst, roi))