# import needed modules
import time
from scipy.optimize import minimize
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import itertools




def sharpeRatio_org(weights,returns_annual, conv_annual):
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = (returns -0.03)/ volatility
    return sharpe

def sharpeRatio(weights, *args):
    returns_annual, cov_annual = args[0],args[1]
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = (returns -0.03)/ volatility
    return 1/sharpe

def constraint1(weights):
    return weights.sum() - 1

def constraint2(weights, *args):
    returns_annual, cov_annual = args[0], args[1]
    returns = np.dot(weights, returns_annual)
    return 1/returns

# <editor-fold desc="data procurement">
symbolsBE=["ABI.BR","ABO.BR","ACCB.BR","ACKB.BR","AED.BR","AGS.BR","AGFB.BR","ANT.BR","ARGX.BR","ASC.BR","ASIT.BR","GEN.BR","ATEB.BR","BALTA.BR","BBV.BR","SANTA.BR","BANI.BR","BAR.BR","BAS.BR","BEAB.BR","BEFB.BR","BEKB.BR","BELR.BR","BELU.BR","BCART.BR","BOEI.BR","BOTHE.BR","BPOST.BR","BNB.BR","BREB.BR","CAMB.BR","CAND.BR","CPINV.BR","CYAD.BR","CENER.BR","CFEB.BR","CHTEX.BR","COMB.BR","CIS.BR","COBH.BR","COFB.BR","COFP2.BR","COLR.BR","CONN.BR","OPTI.BR","DIE.BR","DECB.BR","DTEL.BR","DEXB.BR","DIEG.BR","DISL.BR","EON.BR","ECONB.BR","ELI.BR","ALEMK.BR","ENI.BR","EURN.BR","ALPBS.BR","ALEVA.BR","EVS.BR","EXM.BR","FAGR.BR","FLEX.BR","FLOB.BR","FLUX.BR","FNG.BR","FOU.BR","GBLB.BR","GENK.BR","GIMB.BR","GLOG.BR","GREEN.BR","GROWN.BR","HAMO.BR","HOMI.BR","IBAB.BR","IEP.BR","MCC.BR","IMMOU.BR","IMMO.BR","INCO.BR","INTO.BR","JEN.BR","KBC.BR","KBCA.BR","KEYW.BR","KIN.BR","LEAS.BR","LOTB.BR","LUXA.BR","MDXH.BR","MELE.BR","MSF.BR","MIKO.BR","MITRA.BR","MONT.BR","MOP.BR","MOUR.BR","MEURV.BR","NEU.BR","NEWT.BR","NYR.BR","ONTEX.BR","OBEL.BR","OXUR.BR","PAY.BR","PIC.BR","PROX.BR","QRF.BR","QFG.BR","REC.BR","REI.BR","RES.BR","RET.BR","ENGB.BR","ROU.BR","SAB.BR","SCHD.BR","SEQUA.BR","SHUR.BR","SIA.BR","SIOE.BR","SIP.BR","SMAR.BR","SOF.BR","SOFT.BR","SOLV.BR","SOLB.BR","SPA.BR","SUCR.BR","TIT.BR","TFA.BR","TNET.BR","TERB.BR","TESB.BR","TEXF.BR","TINC.BR","TISN.BR","TUB.BR","UNI.BR","PNSB.BR","UCB.BR","UMI.BR","VAN.BR","VASTB.BR","VGP.BR","VIO.BR","VWA.BR","VWAP.BR","WEB.BR","WDP.BR","WEHB.BR","WOLE.BR","WOLS.BR","XIOR.BR","ZENT.BR","ZEN.BR"]
symbolsBEL20=["ABI.BR","ACKB.BR","APAM.AS","ARGX.BR","BAR.BR","COFB.BR","COLR.BR","GLPG.AS","GBLB.BR","INGA.AMS","KBC.BR","ONTEX.BR","PROX.BR","SOF.BR","SOLB.BR","TNET.BR","UCB.BR","UMI.BR","WDP.BR"]

symbols = symbolsBEL20
print("amount of symbols : {}".format(len(symbols)))

start = datetime(2016, 1, 1)
end = datetime(2017, 12, 31)
table =  None

if not os.path.isfile('yahooDataSet.pkl'):
    datasets = []
    for symbol in symbols:
        print("Fetching data for {}".format(symbol))
        try:
            f = web.DataReader(symbol, 'yahoo',  start, end)
            f['ticker']=np.full(f['Adj Close'].count(),symbol)
            f=f.drop(["High","Low","Open","Volume","Close"],axis=1)
            print("Fetched {}".format(f.shape))
            if f['Adj Close'].count() >=500:
                datasets.append(f)
        except:
            print("Error:{}".format(symbol))
    data = pd.concat(datasets)
    table = data.pivot(columns='ticker')
    table.to_pickle("yahooDataSet.pkl")
else:
    table = pd.read_pickle("yahooDataSet.pkl")

table.columns=table.columns.droplevel()
print(table)
symbolCombinations = itertools.combinations(table.columns,5)
#numberOfCombinations = sum(1 for _ in symbolCombinations)
#print("Number of combinations: {}".format(numberOfCombinations))

port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []
num_portfolios = 5000

# start the clock
start = time. time()
for c,comb in enumerate(symbolCombinations):
    comb2 = list(comb)
    cIndex=[]
    for co in comb2:
        cIndex.append(table.columns.get_loc(co))
    newTable = table.iloc[:,cIndex]
    print("{} -- {}".format(c,comb2))
    num_assets = len(comb2)
    returns_daily = newTable.pct_change()
    returns_annual = returns_daily.mean() * 250

    # get daily and covariance of returns of the stock
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 250

    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = (returns - 0.03) / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}
# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio']
# reorder dataframe columns
df = df[column_order]

# find min Volatility & max sharpe values in the dataframe (df)
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]
returns_opt = 0.43663985263474775
volatility_opt = 0.17178156243122653


plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.scatter(x=volatility_opt, y=returns_opt, c='green', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.savefig('efficiency_curve.pdf')