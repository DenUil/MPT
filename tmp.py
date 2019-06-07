 prevOptPortfolio= None














# compare how previous optimal portfolio does with current dataset
if prevOptPortfolio == None:  # this is the first portfolio optimalisation
    prevOptPortfolio = (optPortTickers, optPortFinalWeights)
    prevOptPortTickers, prevOptPortFinalWeights = None, None
else:
    # recalculate previous portfolio with current trainingset
    prevOptPortTickers, prevOptPortFinalWeights = prevOptPortfolio
    cIndex = []
    for co in prevOptPortTickers:
        cIndex.append(table.columns.get_loc(co))
    # Create a subset of the training data that just has the equities in the combination
    newTable = table.iloc[:, cIndex]
    num_assets = len(prevOptPortTickers)
    # calculate daily and annual returns of the stocks
    returns_daily = newTable.pct_change()
    returns_annual = returns_daily.mean() * 250

    # get daily and covariance of returns of the stock
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 250

    # calculate returns volatility sharpe ratio
    prevReturns = np.dot(prevOptPortFinalWeights, returns_annual)
    prevVolatility = np.sqrt(np.dot(prevOptPortFinalWeights.T, np.dot(cov_annual, prevOptPortFinalWeights)))
    prevSharpe = (prevReturns - 0.03) / prevVolatility

    # store current portfolio for comparison next run
    prevOptPortfolio = (optPortTickers, optPortFinalWeights)







    if prevOptPortTickers != None:
        print("Previous portfolio gave us:")
        print(
            "Max Sharpe Ratio: {} for Portofolio {} with respectively weights {}".format(prevSharpe, prevOptPortTickers,
                                                                                         prevOptPortFinalWeights))
        print("Volatility: {} and return: {}".format(prevVolatility, prevReturns))