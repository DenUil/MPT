from multiprocessing import Process, Queue, Manager
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
from dateutil.relativedelta import *
import math

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

def minimizerThread(q, threadIndex, table, resultList):
    while q.empty():
        time.sleep(0.5)
    maxResult = (0,0,0,0,0)
    while not q.empty():
        tickers = q.get()

        cIndex = []
        for co in tickers:
            cIndex.append(table.columns.get_loc(co))
        newTable = table.iloc[:, cIndex]

        #if q.qsize() % 100 == 0 :
        #    print("Thread{} - Processing #{} of Queue{} length: {}".format(threadIndex,resIndex,threadIndex,q.qsize()))
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
        if(sharpe > maxResult[4]):
            maxResult=(tickers,final_weights,returns,volatility,sharpe)

        #mark task as done
        #q.task_done()

    resultList[threadIndex] = maxResult







if __name__ == "__main__":
    globalStopWatchStart = datetime.now()
    prevOptPortfolio = None
    SymbolsByMonth=[]
    WeightsByMonth=[]
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
                    "INGA.AS", "KBC.BR", "ONTEX.BR", "PROX.BR", "SOF.BR", "SOLB.BR", "TNET.BR", "UCB.BR", "UMI.BR",
                    "WDP.BR"]

    symbolsSET50 = ['SUPER.BK','TRITN.BK','TPIPL.BK','MAX.BK','NUSA.BK','TFG.BK','EVER.BK','AQUA.BK','PF.BK','BLAND.BK','EFORL.BK','SIRI.BK','JSP.BK','BEM.BK','UPA.BK','KTC.BK','JAS.BK','PSTC.BK','CGD.BK','ML.BK','GEL.BK','MACO.BK','WHA.BK','RML.BK','RWI.BK','NMG.BK','TMB.BK','ACC.BK','SGP.BK','TRUE.BK','IRPC.BK','QH.BK','IEC.BK','RS.BK','TWZ.BK','T.BK','GUNKUL.BK','ORI.BK','CHG.BK','ANAN.BK','BSM.BK','TRC.BK','CHO.BK','SPALI.BK','BWG.BK','ITD.BK','TPIPP.BK','NEWS.BK','STPI.BK','NWR.BK']

    symbolsEUROSTOXX30 = ['BMW.DE','DPW.DE','G.MI','INGA.AS','MC.PA','PHIA.AS','AIR.PA','ASML.AS','ENEL.MI','ABI.BR','ORA.PA','ENGI.PA','BAYN.DE','SAN.PA','BN.PA','ITX.MC','BBVA.MC','BNP.PA','ENI.MI','DBK.DE','ALV.DE','AI.PA','SAF.PA','IBE.MC','DTE.DE','CA.PA','SU.PA','OR.PA']


    symbolsAEX =  ['INGA.AS','ABN.AS','PHIA.AS','WKL.AS','RDSA.AS','ASML.AS','KPN.AS','MT.AS','VPK.AS','AGN.AS','RAND.AS','NN.AS','REN.AS','HEIA.AS','IMCD.AS','AALB.AS','DSM.AS','UNA.AS','ASRNL.AS','URW.AS','AD.AS','ADYEN.AS','AKZA.AS','GLPG.AS','TKWY.AS']

    symbolsCAC = ['SGO.PA','MC.PA','AIR.PA','CAP.PA','HO.PA','LR.PA','ORA.PA','ENGI.PA','KER.PA','SAN.PA','EN.PA','BN.PA','FP.PA','ATO.PA','BNP.PA','UG.PA','SU.PA','VIV.PA','OR.PA','ML.PA','AI.PA','FTI.PA','DG.PA','VIE.PA','RI.PA','ACA.PA','CA.PA','GLE.PA','SW.PA','AC.PA']

    symbolsDAX = ['BMW.DE','DAI.DE','MRK.DE','DPW.DE','FRE.DE','CON.DE','VOW3.DE','BAYN.DE','LIN.DE','SAP.DE','MUV2.DE','DB1.DE','EOAN.DE','HEN3.DE','DBK.DE','ALV.DE','VNA.DE','HEI.DE','ADS.DE','BAS.DE','RWE.DE','DTE.DE','SIE.DE','BEI.DE','1COV.DE','FME.DE','IFX.DE','TKA.DE','LHA.DE','WDI.DE']

    symbolsIBEX = ['MRL.MC','GRF.MC','ELE.MC','REE.MC','FER.MC','AMS.MC','ENC.MC','ITX.MC','BBVA.MC','ENG.MC','IAG.MC','ANA.MC','MAP.MC','ACS.MC','COL.MC','MTS.MC','ACX.MC','TL5.MC','SAN.MC','AENA.MC','SAB.MC','BKT.MC','NTGY.MC','TEF.MC','VIS.MC','IBE.MC','CLNX.MC','CABK.MC','SGRE.MC','MEL.MC']

    symbolsATX = ['SBO.VI','OMV.VI','FACC.VI','VER.VI','VOE.VI','TKA.VI','CAI.VI','VIG.VI','ATS.VI','BG.VI','SPI.VI','EBS.VI','RBI.VI','ANDR.VI','WIE.VI','POST.VI','IIA.VI','UQA.VI','DOC.VI','LNZ.VI']

    symbolsOMX = ['ABB.ST','TELIA.ST','NDA-SE.ST','SEB-A.ST','HEXA-B.ST','AZN.ST','SKF-B.ST','SSAB-A.ST','SHB-A.ST','TEL2-B.ST','SAND.ST','SWED-A.ST','ERIC-B.ST','ATCO-A.ST','INVE-B.ST','ATCO-B.ST','ASSA-B.ST','KINV-B.ST','VOLV-B.ST','ELUX-B.ST','SCA-B.ST','SWMA.ST','SKA-B.ST','ALIV-SDB.ST','ALFA.ST','GETI-B.ST','ESSITY-B.ST','BOL.ST-','SECU-B.ST','HM-B.ST']

    symbolsOMXHEX25 = ['NDA-FI.HE','NOKIA.HE','FORTUM.HE','KNEBV.HE','NESTE.HE','STERV.HE','UPM.HE','','ORNBV.HE','KEMIRA.HE','SAMPO.HE','KESKOB.HE','WRT1V.HE','ELISA.HE','METSB.HE','TYRES.HE','TELIA1.HE','HUH1V.HE','OUT1V.HE','VALMT.HE','TIETO.HE','KCR.HE','METSO.HE','OTE1V.HE','CGCBV.HE']

    symbolsList = symbolsBEL20 + symbolsAEX + symbolsDAX + symbolsCAC + symbolsATX + symbolsIBEX + symbolsOMXHEX25 

    symbols = np.unique(symbolsList)

    print ("Found {} unique symbols".format(len(symbols)))
    
    numberOfPermutations = math.factorial(len(symbols))	/ math.factorial(len(symbols)-5)
    
    print ("{} of portfolios to test".format(numberOfPermutations))
    
    #start and end date for the training data
    start = datetime(2014, 11, 11)
    end = datetime(2019, 11, 11)

    #number of threads to start
    num_threads = 16

    #Fetch data
    table = None

    numberOfMonthsPerExpands = 12
    numberOfExpandsPerYear = math.floor(12/numberOfMonthsPerExpands)
    for _ in range(0,numberOfExpandsPerYear):

        startTrainingData = start.strftime("%m/%d/%Y")
        endTrainingData = end.strftime("%m/%d/%Y")

        # start the clock for the processing time
        startStopWatch = time.time()


        #fetch data
        datasets = []
        print("Fetching Equities data for interval: {} - {}".format(startTrainingData,endTrainingData))
        for symbol in symbols:
            try:
                f = web.DataReader(symbol, 'yahoo', start, end)
                f['ticker'] = np.full(f['Adj Close'].count(), symbol)
                f = f.drop(["High", "Low", "Open", "Volume", "Close"], axis=1)
                if f['Adj Close'].count() >= 800:
                    datasets.append(f)
                else:
                    print("{} Failed, not enough datapoints {}".format(symbol,f['Adj Close'].count()))
            except:
                print("Something went wrong with {}".format(symbol))
                pass
        data = pd.concat(datasets)
        table = data.pivot(columns='ticker')
        table.to_pickle("yahooDataSet.pkl")
        table.columns = table.columns.droplevel()
        for col in table.columns:
            table.rename(columns={col: col.replace(".", "_")}, inplace=True)
        #dealing with NaN situation
        table.fillna(method='ffill')
        table.fillna(method='bfill')


        #build all combinations of 5 equities
        print("Building combinations of tickers")
        symbolCombinations = itertools.combinations(table.columns, 5)


        #create new queue
        #initialisation of the result list
        manager = Manager()
        Queues = []
        resultList = manager.list()
        for i in range(num_threads):
            resultList.append(None)
            Queues.append(Queue(maxsize=0))







        queueIndex = 0
        #loop through all symbol combinations we generated
        for comb in symbolCombinations:
            #Create the queue
            Queues[queueIndex].put(comb)
            queueIndex =queueIndex+1
            if(queueIndex==num_threads):
                queueIndex = 0






        # Start the threads; they will wait until they get data via the queue
        threads = []
        for i in range(num_threads):
            worker = Process(target=minimizerThread, args=(Queues[i], i, table, resultList))
            worker.start()
            threads.append(worker)

        print("Waiting for data to be processed by our multiprocessing system...")



        #wait untill all threads are done
        for worker in threads:
            worker.join()

       # take the best of all the threads
        resultListClean = [x if x != None else [0,0,0,0,0] for x in resultList]
        try:
            optimalPortfolio = max(resultListClean,key=itemgetter(4))
            optPortTickers, optPortFinalWeights, optPortReturns, optPortVolatility, optPortSharpe = optimalPortfolio
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
        except:
            print(resultList.index(None))
        endStopWatch = time.time()

        #report data
        print("----------------------------------------------------------------")
        print("----------------------------------------------------------------")
        print("Training data interval: {} - {}".format(startTrainingData, endTrainingData))

        print("Time to completion:{}".format(endStopWatch-startStopWatch))

        print("Max Sharpe Ratio: {} for Portofolio {} with respectively weights {}".format(optPortSharpe,optPortTickers,optPortFinalWeights))
        print("Volatility: {} and return: {}".format(optPortVolatility,optPortReturns))
        if prevOptPortTickers != None:
            print("Previous portfolio gave us:")
            print(
                "Max Sharpe Ratio: {} for Portofolio {} with respectively weights {}".format(prevSharpe,
                                                                                             prevOptPortTickers,
                                                                                             prevOptPortFinalWeights))
            print("Volatility: {} and return: {}".format(prevVolatility, prevReturns))
        print("----------------------------------------------------------------")
        print("----------------------------------------------------------------")
        SymbolsByMonth.append(optPortTickers)
        WeightsByMonth.append(optPortFinalWeights.tolist())
        # Expend dataframe with one
        end = end  + relativedelta(months=numberOfMonthsPerExpands)

    globalStopWatchEnd = datetime.now()
    with open("MPT_results.txt", 'a') as out:
       out.write("Tickers used :{}\n".format(symbols))
       out.write("Time to full completion:{}\n".format(globalStopWatchEnd-globalStopWatchStart))
       out.write("symbolsByMonth={}\n".format(SymbolsByMonth))
       out.write("weightsByMonth={}\n".format(WeightsByMonth))
