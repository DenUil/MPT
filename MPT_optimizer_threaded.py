from queue import Queue
from threading import Thread
from scipy.optimize import minimize
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import itertools
import time
from operator import itemgetter

def computeDataPoints(weights,returns_annual, cov_annual):
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = (returns -0.03)/ volatility
    return returns,volatility,sharpe

def sharpeRatio(weights, *args):
    returns_annual, cov_annual = args[0],args[1]
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = (returns -0.03)/ volatility
    return 1/sharpe

def constraint1(weights):
    return weights.sum() - 1

def minimizerThread(q):
    while q.empty():
        time.sleep(2)

    while not q.empty():
        resIndex, resultList, tickers, newTable = q.get()
        if resIndex % 100 == 0 :
            print("Processing #{} of Queue length: {}".format(resIndex,q.qsize()))
        num_assets = len(tickers)
        returns_daily = newTable.pct_change()
        returns_annual = returns_daily.mean() * 250

        # get daily and covariance of returns of the stock
        cov_daily = returns_daily.cov()
        cov_annual = cov_daily * 250

        # initial guess
        # set random seed for reproduction's sake
        np.random.seed(101)
        weights_0 = np.random.random(num_assets)
        weights_0 /= np.sum(weights_0)

        #arguments for function to minimize
        additional = (returns_annual, cov_annual)
        #bounds
        b = (0.01, 1.0)
        bnds = np.full((num_assets, 2), b)
        #set constraint equality function (sum of all weights should be 1)
        con1 = {'type': 'eq', 'fun': constraint1}
        cons = ([con1])

        #start optimiser
        solution = minimize(sharpeRatio, weights_0, additional, method='SLSQP', bounds=bnds, constraints=cons,
                            tol=0.0001)
        #get results
        final_weights = solution.x

        #compute datapoint from weights
        returns, volatility, sharpe = computeDataPoints(final_weights, returns_annual, cov_annual)

        #put data in resultlist
        resultList[resIndex]=(tickers,final_weights,returns,volatility,sharpe)

        #mark task as done
        q.task_done()








if __name__ == "__main__":

    #Definitions
    #select the tickers of the equities you are interested in
    symbolsBE = ["ABI.BR", "ABO.BR", "ACCB.BR", "ACKB.BR", "AED.BR", "AGS.BR", "AGFB.BR", "ANT.BR", "ARGX.BR", "ASC.BR",
                 "ASIT.BR", "GEN.BR", "ATEB.BR", "BALTA.BR", "BBV.BR", "SANTA.BR", "BANI.BR", "BAR.BR", "BAS.BR",
                 "BEAB.BR", "BEFB.BR", "BEKB.BR", "BELR.BR", "BELU.BR", "BCART.BR", "BOEI.BR", "BOTHE.BR", "BPOST.BR",
                 "BNB.BR", "BREB.BR", "CAMB.BR", "CAND.BR", "CPINV.BR", "CYAD.BR", "CENER.BR", "CFEB.BR", "CHTEX.BR",
                 "COMB.BR", "CIS.BR", "COBH.BR", "COFB.BR", "COFP2.BR", "COLR.BR", "CONN.BR", "OPTI.BR", "DIE.BR",
                 "DECB.BR", "DTEL.BR", "DEXB.BR", "DIEG.BR", "DISL.BR", "EON.BR", "ECONB.BR", "ELI.BR", "ALEMK.BR",
                 "ENI.BR", "EURN.BR", "ALPBS.BR", "ALEVA.BR", "EVS.BR", "EXM.BR", "FAGR.BR", "FLEX.BR", "FLOB.BR",
                 "FLUX.BR", "FNG.BR", "FOU.BR", "GBLB.BR", "GENK.BR", "GIMB.BR", "GLOG.BR", "GREEN.BR", "GROWN.BR",
                 "HAMO.BR", "HOMI.BR", "IBAB.BR", "IEP.BR", "MCC.BR", "IMMOU.BR", "IMMO.BR", "INCO.BR", "INTO.BR",
                 "JEN.BR", "KBC.BR", "KBCA.BR", "KEYW.BR", "KIN.BR", "LEAS.BR", "LOTB.BR", "LUXA.BR", "MDXH.BR",
                 "MELE.BR", "MSF.BR", "MIKO.BR", "MITRA.BR", "MONT.BR", "MOP.BR", "MOUR.BR", "MEURV.BR", "NEU.BR",
                 "NEWT.BR", "NYR.BR", "ONTEX.BR", "OBEL.BR", "OXUR.BR", "PAY.BR", "PIC.BR", "PROX.BR", "QRF.BR",
                 "QFG.BR", "REC.BR", "REI.BR", "RES.BR", "RET.BR", "ENGB.BR", "ROU.BR", "SAB.BR", "SCHD.BR", "SEQUA.BR",
                 "SHUR.BR", "SIA.BR", "SIOE.BR", "SIP.BR", "SMAR.BR", "SOF.BR", "SOFT.BR", "SOLV.BR", "SOLB.BR",
                 "SPA.BR", "SUCR.BR", "TIT.BR", "TFA.BR", "TNET.BR", "TERB.BR", "TESB.BR", "TEXF.BR", "TINC.BR",
                 "TISN.BR", "TUB.BR", "UNI.BR", "PNSB.BR", "UCB.BR", "UMI.BR", "VAN.BR", "VASTB.BR", "VGP.BR", "VIO.BR",
                 "VWA.BR", "VWAP.BR", "WEB.BR", "WDP.BR", "WEHB.BR", "WOLE.BR", "WOLS.BR", "XIOR.BR", "ZENT.BR",
                 "ZEN.BR"]

    symbolsBEL20 = ["ABI.BR", "ACKB.BR", "APAM.AS", "ARGX.BR", "BAR.BR", "COFB.BR", "COLR.BR", "GLPG.AS", "GBLB.BR",
                    "INGA.AMS", "KBC.BR", "ONTEX.BR", "PROX.BR", "SOF.BR", "SOLB.BR", "TNET.BR", "UCB.BR", "UMI.BR",
                    "WDP.BR"]

    symbols = symbolsBEL20

    #start and end date for the training data
    start = datetime(2016, 1, 1)
    end = datetime(2017, 12, 31)

    #number of threads to start
    num_threads = 4

    #Fetch data
    table = None
    if not os.path.isfile('yahooDataSet.pkl'):
        datasets = []
        for symbol in symbols:
            print("Fetching data for {}".format(symbol))
            try:
                f = web.DataReader(symbol, 'yahoo', start, end)
                f['ticker'] = np.full(f['Adj Close'].count(), symbol)
                f = f.drop(["High", "Low", "Open", "Volume", "Close"], axis=1)
                print("Fetched {}".format(f.shape))
                if f['Adj Close'].count() >= 500:
                    datasets.append(f)
            except:
                print("Error:{}".format(symbol))
        data = pd.concat(datasets)
        table = data.pivot(columns='ticker')
        table.to_pickle("yahooDataSet.pkl")
    else:
        table = pd.read_pickle("yahooDataSet.pkl")

    table.columns = table.columns.droplevel()
    print(table)




    #build all combinations of 5 equities
    symbolCombinations = itertools.combinations(table.columns, 5)


    #start the clock for the processing time
    start = time.time()

    #create new queue
    q = Queue(maxsize=0)

    #Start the threads; they will wait until they get data via the queue
    threads = []
    for i in range(num_threads):
        worker = Thread(target=minimizerThread, args=(q,))
        worker.setDaemon(True)
        worker.start()
        threads.append(worker)


    #initialisation of the result list
    resultList = []

    #loop through all symbol combinations we generated
    for c, comb in enumerate(symbolCombinations):
        comb2 = list(comb)
        cIndex = []
        for co in comb2:
            cIndex.append(table.columns.get_loc(co))
        #Create a subset of the training data that just has the equities in the combination
        newTable = table.iloc[:, cIndex]
        num_assets = len(comb2)
        #make a spot available in the result list for the result data
        resultList.append(None)
        #Create the queue
        q.put((c,resultList,comb2,newTable))


    #wait untill all threads are done
    for worker in threads:
        worker.join()

    print(len(resultList))
    optimalPortfolio = max(resultList,key=itemgetter(4))
    optPortTickers, optPortFinalWeights, optPortReturns, optPortVolatility, optPortSharpe = optimalPortfolio
    end = time.time()

    print("Time to completion:{}".format(end-start))

    print("Max Sharpe Ratio: {} for Portofolio {} with respectively weights {}".format(optPortSharpe,optPortTickers,optPortFinalWeights))
    print("Volatility: {} and return: {}".format(optPortVolatility,optPortReturns))
