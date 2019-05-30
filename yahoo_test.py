import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import itertools

symbols=["ABI.BR","ABO.BR","ACPH.BR","ACCB.BR","ACKB.BR","AED.BR","AGS.BR","AGFB.BR","ANT.BR","ARGX.BR","ASC.BR","ASIT.BR","GEN.BR","ATEB.BR","BALTA.BR","BBV.BR","SANTA.BR","BANI.BR","BAR.BR","BAS.BR","BEAB.BR","BEFB.BR","BEKB.BR","BELR.BR","BELU.BR","BCART.BR","BOEI.BR","BOTHE.BR","BPOST.BR","BNB.BR","BREB.BR","CAMB.BR","CAND.BR","CPINV.BR","CYAD.BR","CENER.BR","CFEB.BR","CHTEX.BR","COMB.BR","CIS.BR","COBH.BR","COFB.BR","COFP2.BR","COLR.BR","CONN.BR","OPTI.BR","DIE.BR","DECB.BR","DTEL.BR","DEXB.BR","DIEG.BR","DISL.BR","EON.BR","ECONB.BR","ELI.BR","ALEMK.BR","ENI.BR","EURN.BR","ALPBS.BR","ALEVA.BR","EVS.BR","EXM.BR","FAGR.BR","FLEX.BR","FLOB.BR","FLUX.BR","FNG.BR","FOU.BR","GBLB.BR","GENK.BR","GIMB.BR","GLOG.BR","GREEN.BR","GROWN.BR","HAMO.BR","HOMI.BR","IBAB.BR","IEP.BR","MCC.BR","IMMOU.BR","IMMO.BR","INCO.BR","INTO.BR","JEN.BR","KBC.BR","KBCA.BR","KEYW.BR","KIN.BR","LEAS.BR","LOTB.BR","LUXA.BR","MDXH.BR","MELE.BR","MSF.BR","MIKO.BR","MITRA.BR","MONT.BR","MOP.BR","MOUR.BR","MEURV.BR","NEU.BR","NEWT.BR","NYR.BR","ONTEX.BR","OBEL.BR","OXUR.BR","PAY.BR","PIC.BR","PROX.BR","QRF.BR","QFG.BR","REC.BR","REI.BR","RES.BR","RET.BR","ENGB.BR","ROU.BR","SAB.BR","SCHD.BR","SEQUA.BR","SHUR.BR","SIA.BR","SIOE.BR","SIP.BR","SMAR.BR","SOF.BR","SOFT.BR","SOLV.BR","SOLB.BR","SPA.BR","SUCR.BR","TIT.BR","TFA.BR","TNET.BR","TERB.BR","TESB.BR","TEXF.BR","TINC.BR","TISN.BR","TUB.BR","UNI.BR","PNSB.BR","UCB.BR","UMI.BR","VAN.BR","VASTB.BR","VGP.BR","VIO.BR","VWA.BR","VWAP.BR","WEB.BR","WDP.BR","WEHB.BR","WOLE.BR","WOLS.BR","XIOR.BR","ZENT.BR","ZEN.BR"]

print("amount of symbols : {}".format(len(symbols)))

start = datetime(2009, 5, 24)
end = datetime(2019, 5, 24)
table =  None

if not os.path.isfile('yahooDataSet.pkl'):
    datasets = []
    for symbol in symbols[:5]:
        print("Fetching data for {}".format(symbol))
        f = web.DataReader(symbol, 'yahoo',  start, end)
        f['ticker']=np.full(f['Adj Close'].count(),symbol)
        f=f.drop(["High","Low","Open","Volume","Close"],axis=1)
        print("Fetched {}".format(f.shape))
        datasets.append(f)

    data = pd.concat(datasets)
    table = data.pivot(columns='ticker')
    table.to_pickle("yahooDataSet.pkl")
else:
    table = pd.read_pickle("yahooDataSet.pkl")

print(table)
symbolCombinations = itertools.combinations(symbols,5)


for symbolComb in symbolCombinations:

# calculate daily and annual returns of the stocks
returns_daily = table.pct_change()
returns_annual = returns_daily.mean() * 250

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 250


# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(selected)
num_portfolios = 50000

#set random seed for reproduction's sake
np.random.seed(101)

print(cov_annual)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = (returns -0.03)/ volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(selected):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in selected]

# reorder dataframe columns
df = df[column_order]

# find min Volatility & max sharpe values in the dataframe (df)
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

print(sharpe_portfolio.head())
for stock in selected:
    print("{}=  {}".format(stock+' Weight',sharpe_portfolio.iloc[0][stock+' Weight']))

print("Sharpe optimal point: \n   Avg Return : {} volatility :{} Sharpe ratio:{}".format(sharpe_portfolio.iloc[0]['Returns'],sharpe_portfolio.iloc[0]['Volatility'],sharpe_portfolio.iloc[0]['Sharpe Ratio'],))



 # plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()