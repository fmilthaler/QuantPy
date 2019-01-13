import numpy as np
import pandas as pd
from qpy.pf_quants import weightedMean, weightedStd, sharpeRatio
from qpy.optimisation import optimisePfMC
from qpy.pf_returns import historicalMeanReturn
from qpy.pf_returns import dailyReturns, simpleReturns, dailyLogReturns


class Stock(object):
    '''
    Object that contains information about a stock/fund.
    To initialise the object, it requires a name, information about
    the stock/fund given as one of the following data structures:
     - pandas.Series
     - pandas.DataFrame
    The investment information can contain as little information as its name,
    and the amount invested in it, the column labels must be "Name" and "FMV"
    respectively, but it can also contain more information, such as
     - Year
     - Strategy
     - CCY
     - etc
    It also requires either stock_data, e.g. daily closing prices as a
    pandas.DataFrame or pandas.Series.
    "stock_data" must be given as a DataFrame, and at least one data column
    is required to containing the closing price, hence it is required to
    contain one column label "<stock_name> - Adj. Close" which is used to
    compute the return of investment. However, "stock_data" can contain more
    data in additional columns.
    '''
    def __init__(self, investmentinfo, stock_data):
        self.name = investmentinfo.Name
        self.investmentinfo = investmentinfo
        self.stock_data = stock_data
        # compute expected return and volatility of stock
        self.expectedReturn = self.compExpectedReturn()
        self.volatility = self.compVolatility()
        self.skew = self.__compSkew()
        self.kurtosis = self.__compKurtosis()

    def getInvestmentInfo(self):
        return self.investmentinfo

    # functions to compute quantities
    def compDailyReturns(self):
        return dailyReturns(self.stock_data)

    def compExpectedReturn(self, freq=252):
        '''
        Computes the expected return of the stock.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year
        '''
        return historicalMeanReturn(self.stock_data, freq=freq)

    def compVolatility(self, freq=252):
        '''
        Computes the volatility of the stock.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year
        '''
        return self.compDailyReturns().std() * np.sqrt(freq)

    def __compSkew(self):
        return self.stock_data.skew().values[0]

    def __compKurtosis(self):
        return self.stock_data.kurt().values[0]

    def properties(self):
        # nicely printing out information and quantities of the stock
        string = "-"*50
        string += "\nStock: {}".format(self.name)
        string += "\nExpected return:{:0.3f}".format(
            self.expectedReturn.values[0])
        string += "\nVolatility: {:0.3f}".format(
            self.volatility.values[0])
        string += "\nSkewness: {:0.5f}".format(self.skew)
        string += "\nKurtosis: {:0.5f}".format(self.kurtosis)
        string += "\nInformation:"
        string += "\n"+str(self.investmentinfo.to_frame().transpose())
        string += "\n"
        string += "-"*50
        print(string)

    def __str__(self):
        # print short description
        string = "Contains information about "+str(self.name)+"."
        return string


class Portfolio(object):
    '''
    Object that contains information about a investment portfolio.
    To initialise the object, it does not require any input.
    To fill the portfolio with investment information, the
    function addStock(stock) should be used, in which `stock` is
    a `Stock` object, a pandas.DataFrame of the portfolio investment
    information.
    '''
    def __init__(self):
        # initilisating instance variables
        self.portfolio = pd.DataFrame()
        self.stocks = {}
        self.pf_stock_data = pd.DataFrame()
        self.expectedReturn = None
        self.volatility = None
        self.sharpe = None
        self.skew = None
        self.kurtosis = None
        self.totalinvestment = None

    @property
    def totalinvestment(self):
        return self.__totalinvestment

    @totalinvestment.setter
    def totalinvestment(self, val):
        if (not val is None):
            # treat "None" as initialisation
            if (not isinstance(val, (float, int))):
                raise ValueError("Total investment must be a float or integer.")
            elif (val <= 0):
                raise ValueError("The money to be invested in the portfolio must be > 0.")
            else:
                self.__totalinvestment = val

    def addStock(self, stock):
        # adding stock to dictionary containing all stocks provided
        self.stocks.update({stock.name: stock})
        # adding information of stock to the portfolio
        self.portfolio = self.portfolio.append(
            stock.getInvestmentInfo(),
            ignore_index=True)
        # setting an appropriate name for the portfolio
        self.portfolio.name = "Portfolio information"
        # also add stock data of stock to the dataframe
        self._addStockData(stock.stock_data)

        # compute expected return, volatility and Sharpe ratio of portfolio
        self.totalinvestment = self.portfolio.FMV.sum()
        self.expectedReturn = self.compPfExpectedReturn()
        self.volatility = self.compPfVolatility()
        self.sharpe = self.compPfSharpe()
        self.skew = self.__compPfSkew()
        self.kurtosis = self.__compPfKurtosis()

    def _addStockData(self, df):
        # loop over columns in given dataframe
        for datacol in df.columns:
            cols = len(self.pf_stock_data.columns)
            self.pf_stock_data.insert(loc=cols,
                                      column=datacol,
                                      value=df[datacol].values
                                      )
        # set index correctly
        self.pf_stock_data.set_index(df.index.values, inplace=True)
        # set index name:
        self.pf_stock_data.index.rename('Date', inplace=True)

    def getStock(self, name):
        return self.getStocks()[name]

    def getStocks(self):
        return self.stocks


    # functions to compute quantities
    def compPfSimpleReturns(self):
        '''
        Computes the returns of all stocks in the portfolio.
        price_{t} / price_{t=0}
        '''
        return simpleReturns(self.pf_stock_data)

    def compPfDailyReturns(self):
        '''
        Computes the daily returns (percentage change) of all
        stocks in the portfolio.
        '''
        return dailyReturns(self.pf_stock_data)

    def compPfDailyLogReturns(self):
        '''
        Computes the daily log returns of all stocks in the portfolio.
        '''
        return dailyLogReturns(self.pf_stock_data)

    def compPfMeanReturns(self, freq=252):
        '''
        Computes the mean return based on historical stock price data.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year
        '''
        return historicalMeanReturn(self.pf_stock_data, freq=freq)

    def compPfWeights(self):
        # computes the weights of the stocks in the given portfolio
        # in respect of the total investment
        return self.portfolio['FMV']/self.totalinvestment

    def compPfExpectedReturn(self, freq=252):
        '''
        Computes the expected return of the portfolio.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year
        '''
        pf_return_means = historicalMeanReturn(self.pf_stock_data,
                                               freq=freq)
        weights = self.compPfWeights()
        expectedReturn = weightedMean(pf_return_means.values, weights)
        self.expectedReturn = expectedReturn
        return expectedReturn

    def compPfVolatility(self, freq=252):
        '''
        Computes the volatility of the given portfolio.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year
        '''
        # computing the volatility of a portfolio
        volatility = weightedStd(self.compCovPf(),
                                 self.compPfWeights()) * np.sqrt(freq)
        self.volatility = volatility
        return volatility

    def compCovPf(self):
        # get the covariance matrix of the mean returns of the portfolio
        returns = dailyReturns(self.pf_stock_data)
        return returns.cov()

    def compPfSharpe(self, riskFreeRate=0.005):
        # compute the Sharpe Ratio of the portfolio
        sharpe = sharpeRatio(self.expectedReturn,
                             self.volatility,
                             riskFreeRate)
        self.sharpe = sharpe
        return sharpe

    def __compPfSkew(self):
        return self.pf_stock_data.skew()

    def __compPfKurtosis(self):
        return self.pf_stock_data.kurt()

    # optimising the investments based on volatility and sharpe ratio
    def optimisePortfolio(self,
                          total_investment=None,
                          num_trials=10000,
                          riskFreeRate=0.005,
                          freq=252,
                          plot=True):
        '''
        Optimisation of the portfolio by performing a Monte Carlo simulation.

        Input:
         * total_investment: Float (default: None, which results in the sum of
             FMV of the portfolio information), money to be invested.
         * num_trials: Integer (default: 10000), number of portfolios to be
             computed, each with a random distribution of weights/investments
             in each stock
         * riskFreeRate: Float (default: 0.005), the risk free rate as required
             for the Sharpe Ratio
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year
         * plot: Boolean (default: True), if True, a plot of the Monte Carlo
             simulation is shown
        '''
        # if total_investment is not set, use total FMV of given portfolio
        if (total_investment is None):
            total_investment = self.totalinvestment

        return optimisePfMC(self.pf_stock_data,
                            num_trials=num_trials,
                            total_investment=total_investment,
                            riskFreeRate=riskFreeRate,
                            freq=freq,
                            initial_weights=self.compPfWeights().values,
                            plot=plot)

    def properties(self):
        # nicely printing out information and quantities of the portfolio
        string = "-"*50
        stocknames = self.portfolio.Name.values.tolist()
        string += "\nStocks: {}".format(", ".join(stocknames))
        string += "\nPortfolio expected return: {:0.3f}".format(
            self.expectedReturn)
        string += "\nPortfolio volatility: {:0.3f}".format(
            self.volatility)
        string += "\nPortfolio Sharpe ratio: {:0.3f}".format(
            self.sharpe)
        string += "\nSkewness:"
        string += "\n"+str(self.skew.to_frame().transpose())
        string += "\nKurtosis:"
        string += "\n"+str(self.kurtosis.to_frame().transpose())
        string += "\nInformation:"
        string += "\n"+str(self.portfolio)
        string += "\n"
        string += "-"*50
        print(string)

    def __str__(self):
        # print short description
        string = "Contains information about a portfolio."
        return string


def _correctQuandlRequestStockName(names):
    '''
    This function makes sure that all strings in the given list of
    stock names are leading with "WIKI/" as required by quandl to
    request data.

    Example: If an element of names is "GOOG" (which stands for
    Google), this function modifies the element of names to "WIKI/GOOG".
    '''
    # make sure names is a list of names:
    if (isinstance(names, str)):
        names = [names]
    reqnames = []
    # correct stock names if necessary:
    for name in names:
        if (not name.startswith('WIKI/')):
            name = 'WIKI/'+name
        reqnames.append(name)
    return reqnames


def _quandlRequest(names, start_date=None, end_date=None):
    '''
    This function performs a simple request from quandl and returns
    a DataFrame containing stock data.

    Input:
     * names: List of strings of stock names to be requested
     * start_date (optional): String/datetime of the start date of
         relevant stock data
     * end_date (optional): String/datetime of the end date of
         relevant stock data
    '''
    try:
        import quandl
    except ImportError:
        print("The following package is required:\n - quandl\n"
              + "Please make sure that it is installed.")
    # get correct stock names that quandl.get can request,
    # e.g. "WIKI/GOOG" for Google
    reqnames = _correctQuandlRequestStockName(names)
    return quandl.get(reqnames, start_date=start_date, end_date=end_date)


def _getQuandlDataColumnLabel(stock_name, data_label):
    '''
    Given stock name and label of a data column, this function returns
    the string "<stock_name> - <data_label>" as it can be found in a
    DataFrame returned by quandl.
    '''
    return stock_name+' - '+data_label


def _getStocksDataColumns(stock_data, names, cols):
    ''' This function returns a subset of the given DataFrame stock_data,
        which contains only the data columns as specified in the input cols.

        Input:
         * stock_data: A DataFrame which contains quantities of the stocks
             listed in pf_information
         * names: A string or list of strings, containing the names of the
             stocks, e.g. 'GOOG' for Google.
         * cols: A list of strings of column labels of stock_data to be
             extracted.
        Output:
         * stock_data: A DataFrame which contains only the data columns of
             stock_data as specified in cols.
    '''
    # get correct stock names that quandl get request
    reqnames = _correctQuandlRequestStockName(names)
    # get current column labels and replacement labels
    reqcolnames = []
    for i in range(len(names)):
        for col in cols:
            # differ between dataframe directly from quandl and
            # possibly previously processed dataframe, e.g.
            # read in from disk with slightly modified column labels
            # 1. if <stock_name> in column labels
            if (names[i] in stock_data.columns):
                colname = names[i]
            # 2. if "WIKI/<stock_name> - <col>" in column labels
            elif (_getQuandlDataColumnLabel(reqnames[i], col) in
                  stock_data.columns):
                colname = _getQuandlDataColumnLabel(reqnames[i], col)
            # 3. if "<stock_name> - <col>" in column labels
            elif (_getQuandlDataColumnLabel(names[i], col) in
                  stock_data.columns):
                colname = _getQuandlDataColumnLabel(names[i], col)
            # else, error
            else:
                raise ValueError("Could not find column labels in given "
                                 + "dataframe.")
            # append correct name to list of correct names
            reqcolnames.append(colname)

    stock_data = stock_data.loc[:, reqcolnames]
    # now rename the columns (removing "WIKI/" from column labels):
    newcolnames = {}
    for i in reqcolnames:
        newcolnames.update({i: i.replace('WIKI/', '')})
    stock_data.rename(columns=newcolnames, inplace=True)
    # if only one data column per stock exists, rename column labels
    # to the name of the corresponding stock
    newcolnames = {}
    if (len(cols) == 1):
        for i in range(len(names)):
            newcolnames.update({_getQuandlDataColumnLabel(
                names[i], cols[0]): names[i]})
        stock_data.rename(columns=newcolnames, inplace=True)
    return stock_data


def _buildPortfolioFromQuandl(pf_information,
                              names,
                              start_date=None,
                              end_date=None):
    '''
    Returns a portfolio based on input in form of a list of strings/names
    of stocks.

    Input:
     * pf_information: DataFrame with the required data column
         labels "Name" and "FMV" of the stocks.
     * names: A string or list of strings, containing the names of the
         stocks, e.g. 'GOOG' for Google.
     * start_date (optional): String/datetime start date of stock data to
         be requested through quandl (default: None)
     * end_date (optional): String/datetime end date of stock data to be
         requested through quandl (default: None)
    Output:
     * pf: Instance of Portfolio which contains all the information
         requested by the user.
    '''
    # create an empty portfolio
    pf = Portfolio()
    # request data from quandl:
    stock_data = _quandlRequest(names, start_date, end_date)
    # build portfolio:
    pf = _buildPortfolioFromDf(pf_information, stock_data)
    return pf


def _stocknamesInDataColumns(names, df):
    '''
    Returns True if at least one element of names was found as a column
    label in the dataframe df.
    '''
    return any((name in label for name in names for label in df.columns))


def _buildPortfolioFromDf(pf_information,
                          stock_data,
                          datacolumns=["Adj. Close"]):
    '''
    Returns a portfolio based on input in form of pandas.DataFrame.

    Input:
     * pf_information: DataFrame with the required data column labels
         "Name" and "FMV" of the stocks.
     * stock_data: A DataFrame which contains prices of the stocks
         listed in pf_information
     * datacolumns (optional): A list of strings of data column labels
         to be extracted and returned (default: ["Adj. Close"]).
    Output:
     * pf: Instance of Portfolio which contains all the information
         requested by the user.
    '''
    # make sure stock names are in data dataframe
    if (not _stocknamesInDataColumns(pf_information.Name.values,
                                     stock_data)):
        raise ValueError("Error: None of the provided stock names were"
                         + "found in the provided dataframe.")
    # extract only 'Adj. Close' column from DataFrame:
    stock_data = _getStocksDataColumns(stock_data,
                                       pf_information.Name.values,
                                       datacolumns)
    # building portfolio:
    pf = Portfolio()
    for i in range(len(pf_information)):
        # get name of stock
        name = pf_information.loc[i].Name
        # extract data column(s) of said stock
        stock_stock_data = stock_data.filter(regex=name)
        # if only one data column per stock exists, give dataframe a name
        if (len(datacolumns) == 1):
            stock_stock_data.name = datacolumns[0]
        # create Stock instance and add it to portfolio
        pf.addStock(Stock(pf_information.loc[i],
                          stock_data=stock_stock_data))
    return pf


def _allListEleInOther(l1, l2):
    '''
    Returns True if all elements of list l1 are found in list l2.
    '''
    return all(ele in l2 for ele in l1)


def _anyListEleInOther(l1, l2):
    '''
    Returns True if any element of list l1 is found in list l2.
    '''
    return any(ele in l2 for ele in l1)


def _listComplement(A, B):
    '''
    Returns the relative complement of A in B (also denoted as A\\B)
    '''
    return list(set(B) - set(A))


def buildPortfolio(pf_information, **kwargs):
    '''
    This function builds and returns a portfolio given a set ofinput
    arguments.

    Input:
     * pf_information: This input is always required. DataFrame with
         the required data column labels "Name" and "FMV" of the stocks.
     * names: A string or list of strings, containing the names of the
         stocks, e.g. 'GOOG' for Google.
     * start (optional): String/datetime start date of stock data to be
         requested through quandl (default: None)
     * end (optional): String/datetime end date of stock data to be
         requested through quandl (default: None)
     * stock_data (optional): A DataFrame which contains quantities of
         the stocks listed in pf_information
    Output:
     * pf: Instance of Portfolio which contains all the information
         requested by the user.

    Only the following combinations of inputs are allowed:
     * pf_information, names, start_date (optional), end_date (optional)
     * pf_information, stock_data

    Moreover, the two different ways this function can be used are useful
    for
     1. building a portfolio by pulling data from quandl,
     2. building a portfolio by providing stock data which was obtained
         otherwise, e.g. from data files
    '''
    docstringMsg = "Please read through the docstring, " \
                   "'buildPortfolio.__doc__'."
    inputError = "Error: None of the input arguments {} are allowed " \
                 "in combination with {}. "+docstringMsg
    if (kwargs is None):
        raise ValueError("Error: "+docstringMsg)

    # create an empty portfolio
    pf = Portfolio()

    # list of all valid optional input arguments
    allInputArgs = ['names',
                    'start_date',
                    'end_date',
                    'stock_data']

    # 1. names, start_date, end_date
    allowedInputArgs = ['names',
                        'start_date',
                        'end_date']
    complementInputArgs = _listComplement(allowedInputArgs, allInputArgs)
    if (_allListEleInOther(['names'], kwargs.keys())):
        # check that no input argument conflict arises:
        if (_anyListEleInOther(complementInputArgs, kwargs.keys())):
            raise ValueError(inputError.format(
                complementInputArgs, allowedInputArgs))
        # get portfolio:
        pf = _buildPortfolioFromQuandl(pf_information, **kwargs)

    # 2. stock_data
    allowedInputArgs = ['stock_data']
    complementInputArgs = _listComplement(allowedInputArgs, allInputArgs)
    if (_allListEleInOther(['stock_data'], kwargs.keys())):
        # check that no input argument conflict arises:
        if (_anyListEleInOther(_listComplement(
             allowedInputArgs, allInputArgs), kwargs.keys())):
            raise ValueError(inputError.format(
                complementInputArgs, allowedInputArgs))
        # get portfolio:
        pf = _buildPortfolioFromDf(pf_information, **kwargs)

    return pf
