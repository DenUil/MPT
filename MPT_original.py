# import needed modules
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# get adjusted closing prices of 5 selected companies with Quandl
quandl.ApiConfig.api_key = '4L9pz5K5udKU-emCMi9f'
selected = ['CNP', 'F', 'WMT', 'GE', 'TSLA']
data = quandl.get_table('WIKI/PRICES', ticker = selected,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2014-1-1', 'lte': '2016-12-31' }, paginate=True)

clean = data.set_index('date')
table = clean.pivot(columns='ticker')


# calculate daily and annual returns of the stocks
returns_daily = table.pct_change()
returns_annual = returns_daily.mean() * 250

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 250

# start the clock
start = time. time()

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(selected)
num_portfolios = 100000

#set random seed for reproduction's sake
np.random.seed(101)

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


end = time.time()

print("Time to completion:{}".format(end-start))

print(sharpe_portfolio.head())
for stock in selected:
    print("{}=  {}".format(stock+' Weight',sharpe_portfolio.iloc[0][stock+' Weight']))

print("Sharpe optimal point: \n   Avg Return : {} volatility :{} Sharpe ratio:{}".format(sharpe_portfolio.iloc[0]['Returns'],sharpe_portfolio.iloc[0]['Volatility'],sharpe_portfolio.iloc[0]['Sharpe Ratio'],))

#print("Min Variance point: \n   Avg Return : {} volatility :{}".format(min_variance_port.iloc[0]['Returns'],min_variance_port.iloc[0]['Volatility']))




# from optimiser for comparison
w_opt = np.asarray([0.29709982, 0.01      , 0.40439223, 0.27850794, 0.01      ])
returns_opt = np.dot(w_opt, returns_annual)
volatility_opt = np.sqrt(np.dot(w_opt.T, np.dot(cov_annual, w_opt)))
sharpe_opt = (returns_opt - 0.03) / volatility_opt
print("Sharpe optimizer point: \n   Avg Return : {} volatility :{} Sharpe ratio:{}".format(returns_opt,volatility_opt,sharpe_opt))



 # plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.scatter(x=volatility_opt, y=returns_opt, c='green', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()
