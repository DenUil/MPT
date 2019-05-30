import time
from scipy.optimize import minimize
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import itertools


symbolsByMonth=[['BAR.BR', 'GLPG.AS', 'KBC.BR', 'SOF.BR', 'WDP.BR'],
                ['BAR.BR', 'GLPG.AS', 'KBC.BR', 'SOF.BR', 'WDP.BR'],
                ['BAR.BR', 'GLPG.AS', 'KBC.BR', 'SOF.BR', 'WDP.BR']]
weightsByMonth=[[0.1490478613106863, 0.060218010908534875, 0.17425039633456707, 0.2853146322095054, 0.3311690992367064],
                [0.1753831409474745, 0.05402572696846603, 0.1521966171813208, 0.2813585105847066, 0.3370360043180321],
                [0.1607987909002814, 0.04955418814095064, 0.08799007962379218, 0.3664449205482959, 0.33521202078667994]]

print(len(symbolsByMonth))

indexReference = "^BFX"

start = datetime(2018, 1, 1)
end = datetime(2018, 12, 31)
symbols =  set(x for l in symbolsByMonth for x in l)

print(symbols)

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
table.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
table.columns = table.columns.droplevel()
print("Table Columns {}".format(table.columns))
for col in table.columns:

    table.rename(columns={col:col.replace(".","_")}, inplace=True)

f = web.DataReader(indexReference, 'yahoo',  start, end)
f['ticker']=np.full(f['Adj Close'].count(),indexReference)
indexData=f.drop(["High","Low","Open","Volume","Close"],axis=1)
tableIndex = indexData.pivot(columns='ticker')
tableIndex.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)

#print(tableIndex)
#get the first value of the indexReference at day One
indexRefValue = tableIndex.iloc[0,0]

dataframeToPlot = pd.DataFrame(columns=["Date","portfolio"])
value = indexRefValue * 10


print(table.head())
prevMonth = 0
shares=[]
for date,row in table.iterrows():
    #print("Data {} - row {}".format(date, row['Adj Close']))
    #print("Month as index {}".format(date.month))
    #print(row)

    # select the columns that are relevant based on the selected month in the symbolsByMonth
    # get the correct weights from weights by month
    symbols = symbolsByMonth[date.month-1]
    weights = weightsByMonth[date.month-1]
    #print("Using symbols {}".format(symbols))
    stockPrice = []
    for symbol in symbols:
        stockPrice.append(row[symbol.replace(".","_")])
    #print(stockPrice)

    #should only be calculated once with an new purchase!
    if prevMonth < date.month :
        value = value  - (7.5*len(symbols)*2)
        shares = np.dot(value,weights)/stockPrice
        prevMonth = date.month

    #( weight * value)/ value of the stock on first day
    value = np.dot(shares,stockPrice)


    dict = { 'Date' : date ,
            'portfolio' : value
            }
    #print(dict)
    dataframeToPlot = dataframeToPlot.append(dict,ignore_index=True)


tableIndex['Adj_Close'] = tableIndex['Adj_Close'].apply(lambda x: x*10)
dataframeToPlot.set_index('Date', inplace=True)
dataframeToPlot.index = pd.to_datetime(dataframeToPlot.index)
plt.style.use('seaborn-dark')
ax = dataframeToPlot.plot.line(y='portfolio',c='red', figsize=(10, 8), grid=True)
tableIndex.plot.line(y='Adj_Close', c='blue', ax=ax)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Index Comparison Monthly Modified')
plt.show()
winst = dataframeToPlot['portfolio'][-1]-dataframeToPlot['portfolio'][0]
roi = winst / dataframeToPlot['portfolio'][0]
print("Winst:{} ROI {}".format(winst, roi))