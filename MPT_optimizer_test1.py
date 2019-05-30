# import needed modules
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

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
# get adjusted closing prices of 5 selected companies with Quandl
quandl.ApiConfig.api_key = '4L9pz5K5udKU-emCMi9f'
selected = ['CNP', 'F', 'WMT', 'GE', 'TSLA']
num_assets = len(selected)
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
# </editor-fold>

# start the clock
start = time. time()

#initial guess
#set random seed for reproduction's sake
np.random.seed(101)
weights_0 = np.random.random(num_assets)
weights_0 /= np.sum(weights_0)

additional = (returns_annual,cov_annual)
b=(0.01,1.0)
bnds=np.full((num_assets,2),b)
con1 = {'type':'eq', 'fun': constraint1}
#con2 = {'type':'ineq', 'fun': constraint2, 'args':additional}
#cons = ([con1,con2])
cons = ([con1])
solution = minimize(sharpeRatio,weights_0,additional,method='SLSQP',bounds=bnds,constraints=cons, tol=0.0001)
end = time.time()

print("Time to completion:{}".format(end-start))
print(solution)

final_weights = solution.x

print('Final Objective: ' + str(sharpeRatio_org(final_weights , returns_annual,cov_annual)))

# print solution
print('Solution')
print('x1 = ' + str(final_weights[0]))
print('x2 = ' + str(final_weights[1]))
print('x3 = ' + str(final_weights[2]))
print('x4 = ' + str(final_weights[3]))
print('x4 = ' + str(final_weights[4]))


print(final_weights.sum())
