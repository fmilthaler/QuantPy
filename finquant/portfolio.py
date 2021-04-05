"""This module is the **core** of `FinQuant`. It provides

- a public class ``Stock`` that holds and calculates quantities of a single stock,
- a public class ``Portfolio`` that holds and calculates quantities of a financial
  portfolio, which is a collection of Stock instances.
- a public function ``build_portfolio()`` that automatically constructs and returns
  an instance of ``Portfolio`` and instances of ``Stock``. The relevant stock
  data is either retrieved through `quandl`/`yfinance` or provided by the user as a
  ``pandas.DataFrame`` (after loading it manually from disk/reading from file).
  For an example on how to use it, please read the corresponding docstring,
  or have a look at the examples in the sub-directory ``example``.

The classes ``Stock`` and ``Portfolio`` are designed to easily manage your
financial portfolio, and make the most common quantitative calculations, such as:

- cumulative returns of the portfolio's stocks
- daily returns of the portfolio's stocks (daily percentage change),
- daily log returns of the portfolio's stocks,
- Expected (annualised) Return,
- Volatility,
- Sharpe Ratio,
- skewness of the portfolio's stocks,
- Kurtosis of the portfolio's stocks,
- the portfolio's covariance matrix.

Furthermore, the constructed portfolio can be optimised for

- minimum Volatility,
- maximum Sharpe Ratio
- minimum Volatility for a given Expected Return
- maximum Sharpe Ratio for a given target Volatility

by either performing a numerical computation to solve a minimisation problem, or by performing a Monte Carlo simulation of `n` trials.
The former should be the preferred method for reasons of computational effort
and accuracy. The latter is only included for the sake of completeness.

Finally, functions are implemented to generated the following plots:

- Monte Carlo run to find optimal portfolio(s)
- Efficient Frontier
- Portfolio with the minimum Volatility based a numerical optimisation
- Portfolio with the maximum Sharpe Ratio based on a numerical optimisation
- Portfolio with the minimum Volatility for a given Expected Return based
  on a numerical optimisation
- Portfolio with the maximum Sharpe Ratio for a given target Volatility
  based on a numerical optimisation
- Individual stocks of the portfolio (Expected Return over Volatility)
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from finquant.quants import weighted_mean, weighted_std, sharpe_ratio, sortino_ratio, downside_risk
from finquant.returns import historical_mean_return
from finquant.returns import daily_returns, cumulative_returns
from finquant.returns import daily_log_returns
from finquant.efficient_frontier import EfficientFrontier
from finquant.monte_carlo import MonteCarloOpt


class Stock(object):
    """Object that contains information about a stock/fund.
    To initialise the object, it requires a name, information about
    the stock/fund given as one of the following data structures:

    - ``pandas.Series``
    - ``pandas.DataFrame``

    The investment information can contain as little information as its name,
    and the amount invested in it, the column labels must be ``Name`` and ``Allocation``
    respectively, but it can also contain more information, such as

    - Year
    - Strategy
    - CCY
    - etc.

    It also requires either data, e.g. daily closing prices as a
    ``pandas.DataFrame`` or ``pandas.Series``.
    ``data`` must be given as a ``pandas.DataFrame``, and at least one data column
    is required to containing the closing price, hence it is required to
    contain one column label ``<stock_name> - Adj. Close`` which is used to
    compute the return of investment. However, ``data`` can contain more
    data in additional columns.
    """

    def __init__(self, investmentinfo, data):
        """
        :Input:
         :investmentinfo: ``pandas.DataFrame`` of investment information
         :data: ``pandas.DataFrame`` of stock price
        """
        self.name = investmentinfo.Name
        self.investmentinfo = investmentinfo
        self.data = data
        # compute expected return and volatility of stock
        self.expected_return = self.comp_expected_return()
        self.volatility = self.comp_volatility()
        self.skew = self._comp_skew()
        self.kurtosis = self._comp_kurtosis()

    # functions to compute quantities
    def comp_daily_returns(self):
        """Computes the daily returns (percentage change).
        See ``finquant.returns.daily_returns``.
        """
        return daily_returns(self.data)

    def comp_expected_return(self, freq=252):
        """Computes the Expected Return of the stock.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year

        :Output:
         :expected_return: Expected Return of stock.
        """
        return historical_mean_return(self.data, freq=freq)

    def comp_volatility(self, freq=252):
        """Computes the Volatility of the stock.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year

        :Output:
         :volatility: Volatility of stock.
        """
        return self.comp_daily_returns().std() * np.sqrt(freq)

    def _comp_skew(self):
        """Computes and returns the skewness of the stock."""
        return self.data.skew().values[0]

    def _comp_kurtosis(self):
        """Computes and returns the Kurtosis of the stock."""
        return self.data.kurt().values[0]

    def properties(self):
        """Nicely prints out the properties of the stock: Expected Return,
        Volatility, Skewness, Kurtosis as well as the ``Allocation`` (and other
        information provided in investmentinfo.)
        """
        # nicely printing out information and quantities of the stock
        string = "-" * 50
        string += "\nStock: {}".format(self.name)
        string += "\nExpected Return:{:0.3f}".format(self.expected_return.values[0])
        string += "\nVolatility: {:0.3f}".format(self.volatility.values[0])
        string += "\nSkewness: {:0.5f}".format(self.skew)
        string += "\nKurtosis: {:0.5f}".format(self.kurtosis)
        string += "\nInformation:"
        string += "\n" + str(self.investmentinfo.to_frame().transpose())
        string += "\n"
        string += "-" * 50
        print(string)

    def __str__(self):
        # print short description
        string = "Contains information about " + str(self.name) + "."
        return string


class Portfolio(object):
    """Object that contains information about a investment portfolio.
    To initialise the object, it does not require any input.
    To fill the portfolio with investment information, the
    function ``add_stock(stock)`` should be used, in which ``stock`` is
    an object of ``Stock``.
    """

    def __init__(self):
        """Initiates ``Portfolio``."""
        # initilisating instance variables
        self.portfolio = pd.DataFrame()
        self.stocks = {}
        self.data = pd.DataFrame()
        self.expected_return = None
        self.volatility = None
        self.downside_risk = None
        self.sharpe = None
        self.sortino = None
        self.skew = None
        self.kurtosis = None
        self.totalinvestment = None
        self.risk_free_rate = 0.005
        self.freq = 252
        # instance variables for Efficient Frontier and
        # Monte Carlo optimisations
        self.ef = None
        self.mc = None

    @property
    def totalinvestment(self):
        return self.__totalinvestment

    @totalinvestment.setter
    def totalinvestment(self, val):
        if val is not None:
            # treat "None" as initialisation
            if not isinstance(val, (float, int)):
                raise ValueError("Total investment must be a float or integer.")
            elif val <= 0:
                raise ValueError(
                    "The money to be invested in the portfolio must be > 0."
                )
            else:
                self.__totalinvestment = val

    @property
    def freq(self):
        return self.__freq

    @freq.setter
    def freq(self, val):
        if not isinstance(val, int):
            raise ValueError("Time window/frequency must be an integer.")
        elif val <= 0:
            raise ValueError("freq must be > 0.")
        else:
            self.__freq = val
            # now that this changed, update other quantities
            self._update()

    @property
    def risk_free_rate(self):
        return self.__risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, val):
        if not isinstance(val, (float, int)):
            raise ValueError("Risk free rate must be a float or an integer.")
        else:
            self.__risk_free_rate = val
            # now that this changed, update other quantities
            self._update()

    def add_stock(self, stock):
        """Adds a stock of type ``Stock`` to the portfolio. Each time ``add_stock``
        is called, the following instance variables are updated:

        - ``portfolio``: ``pandas.DataFrame``, adds a column with information from ``stock``
        - ``stocks``: ``dictionary``, adds an entry for ``stock``
        - ``data``: ``pandas.DataFrame``, adds a column of stock prices from ``stock``

        Also, the following instance variables are (re-)computed:

        - ``expected_return``: Expected Return of the portfolio
        - ``volatility``: Volatility of the portfolio
        - ``sharpe``: Sharpe Ratio of the portfolio
        - ``skew``: Skewness of the portfolio's stocks
        - ``kurtosis``: Kurtosis of the portfolio's stocks

        :Input:
         :stock: an object of ``Stock``
        """
        # adding stock to dictionary containing all stocks provided
        self.stocks.update({stock.name: stock})
        # adding information of stock to the portfolio
        self.portfolio = self.portfolio.append(stock.investmentinfo, ignore_index=True)
        # setting an appropriate name for the portfolio
        self.portfolio.name = "Allocation of stocks"
        # also add stock data of stock to the dataframe
        self._add_stock_data(stock.data)

        # update quantities of portfolio
        self._update()

    def _add_stock_data(self, df):
        # loop over columns in given dataframe
        for datacol in df.columns:
            cols = len(self.data.columns)
            self.data.insert(loc=cols, column=datacol, value=df[datacol].values)
        # set index correctly
        self.data.set_index(df.index.values, inplace=True)
        # set index name:
        self.data.index.rename("Date", inplace=True)

    def _update(self):
        # sanity check (only update values if none of the below is empty):
        if not (self.portfolio.empty or self.stocks == {} or self.data.empty):
            self.totalinvestment = self.portfolio.Allocation.sum()
            self.expected_return = self.comp_expected_return(freq=self.freq)
            self.volatility = self.comp_volatility(freq=self.freq)
            self.downside_risk = self.comp_downside_risk(freq=self.freq)
            self.sharpe = self.comp_sharpe()
            self.sortino = self.comp_sortino()
            self.skew = self._comp_skew()
            self.kurtosis = self._comp_kurtosis()

    def get_stock(self, name):
        """Returns the instance of ``Stock`` with name ``name``.

        :Input:
         :name: ``string`` of the name of the stock that is returned. Must match
             one of the labels in the dictionary ``self.stocks``.

        :Output:
         :stock: instance of ``Stock``.
        """
        return self.stocks[name]

    def comp_cumulative_returns(self):
        """Computes the cumulative returns of all stocks in the
        portfolio.
        See ``finquant.returns.cumulative_returns``.

        :Output:
         :ret: a ``pandas.DataFrame`` of cumulative returns of given stock prices.
        """
        return cumulative_returns(self.data)

    def comp_daily_returns(self):
        """Computes the daily returns (percentage change) of all
        stocks in the portfolio. See ``finquant.returns.daily_returns``.

        :Output:
         :ret: a ``pandas.DataFrame`` of daily percentage change of Returns
             of given stock prices.
        """
        return daily_returns(self.data)

    def comp_daily_log_returns(self):
        """Computes the daily log returns of all stocks in the portfolio.
        See ``finquant.returns.daily_log_returns``.

        :Output:
         :ret: a ``pandas.DataFrame`` of log Returns
        """
        return daily_log_returns(self.data)

    def comp_mean_returns(self, freq=252):
        """Computes the mean returns based on historical stock price data.
        See ``finquant.returns.historical_mean_return``.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year.

        :Output:
         :ret: a ``pandas.DataFrame`` of historical mean Returns.
        """
        return historical_mean_return(self.data, freq=freq)

    def comp_stock_volatility(self, freq=252):
        """Computes the Volatilities of all the stocks individually

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year.

        :Output:
         :volatilies: ``pandas.DataFrame`` with the individual Volatilities of all stocks
             of the portfolio.
        """
        if not isinstance(freq, int):
            raise ValueError("freq is expected to be an integer.")
        return self.comp_daily_returns().std() * np.sqrt(freq)

    def comp_weights(self):
        """Computes and returns a ``pandas.Series`` of the weights/allocation
        of the stocks of the portfolio.

        :Output:
         :weights: a ``pandas.Series`` with weights/allocation of all stocks
             within the portfolio.
        """
        # computes the weights of the stocks in the given portfolio
        # in respect of the total investment
        return self.portfolio["Allocation"] / self.totalinvestment

    def comp_expected_return(self, freq=252):
        """Computes the Expected Return of the portfolio.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year.

        :Output:
         :expected_return: ``float`` the Expected Return of the portfolio.
        """
        if not isinstance(freq, int):
            raise ValueError("freq is expected to be an integer.")
        pf_return_means = historical_mean_return(self.data, freq=freq)
        weights = self.comp_weights()
        expected_return = weighted_mean(pf_return_means.values, weights)
        self.expected_return = expected_return
        return expected_return

    def comp_volatility(self, freq=252):
        """Computes the Volatility of the given portfolio.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year.

        :Output:
         :volatility: ``float`` the Volatility of the portfolio.
        """
        if not isinstance(freq, int):
            raise ValueError("freq is expected to be an integer.")
        # computing the volatility of a portfolio
        volatility = weighted_std(self.comp_cov(), self.comp_weights()) * np.sqrt(freq)
        self.volatility = volatility
        return volatility

    def comp_downside_risk(self, freq=252):
        """Computes the downside risk of the portfolio.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year

        :Output:
         :volatility: Downside risk of stock.
        """
        return downside_risk(self.data, self.comp_weights(), self.risk_free_rate) * np.sqrt(freq)


    def comp_cov(self):
        """Compute and return a ``pandas.DataFrame`` of the covariance matrix
        of the portfolio.

        :Output:
         :cov: a ``pandas.DataFrame`` of the covariance matrix of the portfolio.
        """
        # get the covariance matrix of the mean returns of the portfolio
        returns = daily_returns(self.data)
        return returns.cov()

    def comp_sharpe(self):
        """Compute and return the Sharpe Ratio of the portfolio.

        :Output:
         :sharpe: ``float``, the Sharpe Ratio of the portfolio
        """
        # compute the Sharpe Ratio of the portfolio
        sharpe = sharpe_ratio(
            self.expected_return, self.volatility, self.risk_free_rate
        )
        self.sharpe = sharpe
        return sharpe

    def comp_sortino(self, freq=252):
        """Compute and return the Sortino Ratio of the portfolio

        :Output:
         :sortino: ``float``, the Sortino Ratio of the portfolio
         May be NaN if the portoflio outperformed the risk free rate at every point
        """
        return sortino_ratio(self.expected_return, self.downside_risk, self.risk_free_rate)
         
    def _comp_skew(self):
        """Computes and returns the skewness of the stocks in the portfolio."""
        return self.data.skew()

    def _comp_kurtosis(self):
        """Computes and returns the Kurtosis of the stocks in the portfolio."""
        return self.data.kurt()

    # optimising the investments with the efficient frontier class
    def _get_ef(self):
        """If self.ef does not exist, create and return an instance of
        finquant.efficient_frontier.EfficientFrontier, else, return the
        existing instance.
        """
        if self.ef is None:
            # create instance of EfficientFrontier
            self.ef = EfficientFrontier(
                self.comp_mean_returns(freq=1),
                self.comp_cov(),
                risk_free_rate=self.risk_free_rate,
                freq=self.freq,
            )
        return self.ef

    def ef_minimum_volatility(self, verbose=False):
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.minimum_volatility``.

        Finds the portfolio with the minimum Volatility.

        :Input:
         :verbose: ``boolean`` (default= ``False``), whether to print out properties
             or not.

        :Output:
         :df_weights: a ``pandas.DataFrame`` of weights/allocation of stocks within
             the optimised portfolio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self._get_ef()
        # perform optimisation
        opt_weights = ef.minimum_volatility()
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_maximum_sharpe_ratio(self, verbose=False):
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.maximum_sharpe_ratio``.

        Finds the portfolio with the maximum Sharpe Ratio, also called the
        tangency portfolio.

        :Input:
         :verbose: ``boolean`` (default= ``False``), whether to print out properties
             or not.

        :Output:
         :df_weights: a ``pandas.DataFrame`` of weights/allocation of stocks within
             the optimised portfolio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self._get_ef()
        # perform optimisation
        opt_weights = ef.maximum_sharpe_ratio()
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_efficient_return(self, target, verbose=False):
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.efficient_return``.

        Finds the portfolio with the minimum Volatility for a given target return.

        :Input:
         :target: ``float``, the target return of the optimised portfolio.
         :verbose: ``boolean`` (default= ``False``), whether to print out properties
             or not.

        :Output:
         :df_weights: a ``pandas.DataFrame`` of weights/allocation of stocks within
             the optimised portfolio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self._get_ef()
        # perform optimisation
        opt_weights = ef.efficient_return(target)
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_efficient_volatility(self, target, verbose=False):
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.efficient_volatility``.

        Finds the portfolio with the maximum Sharpe Ratio for a given
        target Volatility.

        :Input:
         :target: ``float``, the target Volatility of the optimised portfolio.
         :verbose: ``boolean`` (default= ``False``), whether to print out properties
             or not.

        :Output:
         :df_weights: a ``pandas.DataFrame`` of weights/allocation of stocks within
             the optimised portfolio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self._get_ef()
        # perform optimisation
        opt_weights = ef.efficient_volatility(target)
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_efficient_frontier(self, targets=None):
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.efficient_frontier``.

        Gets portfolios for a range of given target Returns.
        If no targets were provided, the algorithm will find the minimum
        and maximum Returns of the portfolio's individual stocks, and set
        the target range according to those values.
        Results in the Efficient Frontier.

        :Input:
         :targets: ``list``/``numpy.ndarray`` (default: ``None``) of ``floats``,
             range of target Returns.

        :Output:
         :efrontier: ``numpy.ndarray`` of (Volatility, Return) values.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self._get_ef()
        # perform optimisation
        efrontier = ef.efficient_frontier(targets)
        return efrontier

    def ef_plot_efrontier(self):
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.plot_efrontier``.

        Plots the Efficient Frontier."""
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self._get_ef()
        # plot efficient frontier
        ef.plot_efrontier()

    def ef_plot_optimal_portfolios(self):
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.plot_optimal_portfolios``.

        Plots markers of the optimised portfolios for

        - minimum Volatility, and
        - maximum Sharpe Ratio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self._get_ef()
        # plot efficient frontier
        ef.plot_optimal_portfolios()

    # optimising the investments with the efficient frontier class
    def _get_mc(self, num_trials=1000):
        """If self.mc does not exist, create and return an instance of
        finquant.monte_carlo.MonteCarloOpt, else, return the existing instance.
        """
        if self.mc is None:
            # create instance of MonteCarloOpt
            self.mc = MonteCarloOpt(
                self.comp_daily_returns(),
                num_trials=num_trials,
                risk_free_rate=self.risk_free_rate,
                freq=self.freq,
                initial_weights=self.comp_weights().values,
            )
        return self.mc

    # optimising the investments by performing a Monte Carlo run
    # based on volatility and sharpe ratio
    def mc_optimisation(self, num_trials=1000):
        """Interface to
        ``finquant.monte_carlo.MonteCarloOpt.optimisation``.

        Optimisation of the portfolio by performing a Monte Carlo
        simulation.

        :Input:
         :num_trials: ``int`` (default: ``1000``), number of portfolios to be
             computed, each with a random distribution of weights/allocation
             in each stock.

        :Output:
         :opt_w: ``pandas.DataFrame`` with optimised investment strategies for maximum
             Sharpe Ratio and minimum Volatility.
         :opt_res: ``pandas.DataFrame`` with Expected Return, Volatility and Sharpe Ratio
             for portfolios with minimum Volatility and maximum Sharpe Ratio.
        """
        # dismiss previous instance of mc, as we are performing a new MC optimisation:
        self.mc = None
        # get instance of MonteCarloOpt
        mc = self._get_mc(num_trials)
        opt_weights, opt_results = mc.optimisation()
        return opt_weights, opt_results

    def mc_plot_results(self):
        """Plots the results of the Monte Carlo run, with all of the randomly
        generated weights/portfolios, as well as markers for the portfolios with the

        - minimum Volatility, and
        - maximum Sharpe Ratio.
        """
        # get instance of MonteCarloOpt
        mc = self._get_mc()
        mc.plot_results()

    def mc_properties(self):
        """Calculates and prints out Expected annualised Return,
        Volatility and Sharpe Ratio of optimised portfolio.
        """
        # get instance of MonteCarloOpt
        mc = self._get_mc()
        mc.properties()

    def plot_stocks(self, freq=252):
        """Plots the Expected annual Returns over annual Volatility of
        the stocks of the portfolio.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year.
        """
        # annual mean returns of all stocks
        stock_returns = self.comp_mean_returns(freq=freq)
        stock_volatility = self.comp_stock_volatility(freq=freq)
        # adding stocks of the portfolio to the plot
        # plot stocks individually:
        plt.scatter(stock_volatility, stock_returns, marker="o", s=100, label="Stocks")
        # adding text to stocks in plot:
        for i, txt in enumerate(stock_returns.index):
            plt.annotate(
                txt,
                (stock_volatility[i], stock_returns[i]),
                xytext=(10, 0),
                textcoords="offset points",
                label=i,
            )
            plt.legend()

    def properties(self):
        """Nicely prints out the properties of the portfolio:

        - Expected Return,
        - Volatility,
        - Sharpe Ratio,
        - skewness,
        - Kurtosis

        as well as the allocation of the stocks across the portfolio.
        """
        # nicely printing out information and quantities of the portfolio
        string = "-" * 70
        stocknames = self.portfolio.Name.values.tolist()
        string += "\nStocks: {}".format(", ".join(stocknames))
        string += "\nTime window/frequency: {}".format(self.freq)
        string += "\nRisk free rate: {}".format(self.risk_free_rate)
        string += "\nPortfolio Expected Return: {:0.3f}".format(self.expected_return)
        string += "\nPortfolio Volatility: {:0.3f}".format(self.volatility)
        string += "\nPortfolio Downside Risk: {:0.3f}".format(self.downside_risk)
        string += "\nPortfolio Sharpe Ratio: {:0.3f}".format(self.sharpe)
        string += "\nPortfolio Sortino Ratio: {:0.3f}".format(self.sortino)
        string += "\n\nSkewness:"
        string += "\n" + str(self.skew.to_frame().transpose())
        string += "\n\nKurtosis:"
        string += "\n" + str(self.kurtosis.to_frame().transpose())
        string += "\n\nInformation:"
        string += "\n" + str(self.portfolio)
        string += "\n"
        string += "-" * 70
        print(string)

    def __str__(self):
        # print short description
        string = "Contains information about a portfolio."
        return string


def _correct_quandl_request_stock_name(names):
    """If given input argument is of type string,
    this function converts it to a list, assuming the input argument
    is only one stock name.
    """
    # make sure names is a list of names:
    if isinstance(names, str):
        names = [names]
    return names


def _quandl_request(names, start_date=None, end_date=None):
    """This function performs a simple request from `quandl` and returns
    a ``pandas.DataFrame`` containing stock data.

    :Input:
     :names: List of strings of stock names to be requested
     :start_date (optional): String/datetime of the start date of
         relevant stock data.
     :end_date (optional): String/datetime of the end date of
         relevant stock data.
    """
    try:
        import quandl
    except ImportError:
        print(
            "The following package is required:\n - `quandl`\n"
            + "Please make sure that it is installed."
        )
    # get correct stock names that quandl.get can request,
    # e.g. "WIKI/GOOG" for Google
    reqnames = _correct_quandl_request_stock_name(names)
    try:
        resp = quandl.get(reqnames, start_date=start_date, end_date=end_date)
    except Exception:
        errormsg = (
            "Error during download of stock data from Quandl.\n"
            + "Make sure all the requested stock names/tickers are "
            + "supported by Quandl."
        )
        raise Exception(errormsg)
    return resp


def _yfinance_request(names, start_date=None, end_date=None):
    """This function performs a simple request from Yahoo Finance
    (using `yfinance`) and returns a ``pandas.DataFrame``
    containing stock data.

    :Input:
     :names: List of strings of stock names to be requested
     :start_date (optional): String/datetime of the start date of
         relevant stock data.
     :end_date (optional): String/datetime of the end date of
         relevant stock data.
    """
    try:
        import yfinance as yf
    except ImportError:
        print(
            "The following package is required:\n - `yfinance`\n"
            + "Please make sure that it is installed."
        )
    # yfinance does not exit safely if start/end date were not given correctly:
    # this step is not required for quandl as it handles this exception properly
    try:
        import datetime

        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ImportError:
        print(
            "The following package is required:\n - `datetime`\n"
            + "Please make sure that it is installed."
        )
    except Exception:
        raise Exception("Please provide valid values for <start_date> and <end_date>")

    # unlike quandl, yfinance does not have a prefix in front of the ticker
    # thus we do not need to correct them
    try:
        resp = yf.download(names, start=start_date, end=end_date)
        if not isinstance(resp.columns, pd.MultiIndex) and len(names) > 0:
            # for single stock must make the dataframe multiindex
            stock_tuples = [(col, names[0]) for col in list(resp.columns)]
            resp.columns = pd.MultiIndex.from_tuples(stock_tuples)
    except Exception:
        raise Exception(
            "Error during download of stock data from Yahoo Finance with `yfinance`."
        )
    return resp


def _get_quandl_data_column_label(stock_name, data_label):
    """Given stock name and label of a data column, this function returns
    the string "<stock_name> - <data_label>" as it can be found in a
    ``pandas.DataFrame`` returned by `quandl`.
    """
    return stock_name + " - " + data_label


def _get_stocks_data_columns(data, names, cols):
    """This function returns a subset of the given ``pandas.DataFrame`` data, which
    contains only the data columns as specified in the input cols.

    :Input:
     :data: A ``pandas.DataFrame`` which contains quantities of the stocks
         listed in pf_allocation.
     :names: A string or list of strings, containing the names of the
         stocks, e.g. 'WIKI/GOOG' for Google.
     :cols: A list of strings of column labels of data to be extracted.
         Currently only one column per stock is supported.

    :Output:
     :data: A ``pandas.DataFrame`` which contains only the data columns of
         data as specified in cols.
    """
    # get correct stock names that quandl get request
    reqnames = _correct_quandl_request_stock_name(names)
    # get current column labels and replacement labels
    reqcolnames = []
    # if dataframe is of type multiindex, also get first level colname
    firstlevel_colnames = []
    for i in range(len(names)):
        for col in cols:
            # differ between dataframe directly from quandl and
            # possibly previously processed dataframe, e.g.
            # read in from disk with slightly modified column labels
            # 1. if <stock_name> in column labels
            if names[i] in data.columns:
                colname = names[i]
            # 2. if "WIKI/<stock_name> - <col>" in column labels
            elif _get_quandl_data_column_label(reqnames[i], col) in data.columns:
                colname = _get_quandl_data_column_label(reqnames[i], col)
            # 3. if "<stock_name> - <col>" in column labels
            elif _get_quandl_data_column_label(names[i], col) in data.columns:
                colname = _get_quandl_data_column_label(names[i], col)
            # if column labels is of type multiindex, and the "Adj Close" is in
            # first level labels, we assume the dataframe comes from yfinance:
            elif isinstance(data.columns, pd.MultiIndex):
                # alter col for yfinance, as it returns column labels without '.'
                col = col.replace(".", "")
                if col in data.columns:
                    if not col in firstlevel_colnames:
                        firstlevel_colnames.append(col)
                    if names[i] in data[col].columns:
                        colname = names[i]
                    else:  # error, it must find names[i] on second level of column header
                        raise ValueError(
                            "Could not find column labels in second level of MultiIndex pd.DataFrame"
                        )
            # else, error
            else:
                raise ValueError("Could not find column labels in given dataframe.")
            # append correct name to list of correct names
            reqcolnames.append(colname)

    # if data comes from yfinance, it is a multiindex dataframe:
    if isinstance(data.columns, pd.MultiIndex):
        if not len(firstlevel_colnames) == 1:
            raise ValueError(
                "Sorry, for now only one value/quantity per Stock is supported."
            )
        data = data[firstlevel_colnames[0]].loc[:, reqcolnames]
    else:
        # if it comes from quandl, it is not of type multiindex
        data = data.loc[:, reqcolnames]

    # if only one data column per stock exists, rename column labels
    # to the name of the corresponding stock
    newcolnames = {}
    if len(cols) == 1:
        for i in range(len(names)):
            newcolnames.update(
                {_get_quandl_data_column_label(names[i], cols[0]): names[i]}
            )
        data.rename(columns=newcolnames, inplace=True)
    return data


def _build_portfolio_from_api(
    names, pf_allocation=None, start_date=None, end_date=None, data_api="quandl"
):
    """Returns a portfolio based on input in form of a list of strings/names
    of stocks.

    :Input:
     :names: A string or list of strings, containing the names of the
         stocks, e.g. 'GOOG' for Google.
     :pf_allocation (optional): ``pandas.DataFrame`` with the required data column
         labels ``Name`` and ``Allocation`` of the stocks.
     :start_date (optional): String/datetime start date of stock data to
         be requested through `quandl`/`yfinance` (default: None)
     :end_date (optional): String/datetime end date of stock data to be
         requested through `quandl`/`yfinance` (default: None)
     :data_api: (optional) A ``string`` (default: ``quandl``) which determines how to
         obtain stock prices, if data is not provided by the user. Valid values:
         - ``quandl`` (Python package/API to `Quandl`)
         - ``yfinance`` (Python package formerly known as ``fix-yahoo-finance``)

    :Output:
     :pf: Instance of Portfolio which contains all the information
         requested by the user.
    """
    # create an empty portfolio
    pf = Portfolio()
    # request data from service:
    if data_api == "yfinance":
        data = _yfinance_request(names, start_date, end_date)
    elif data_api == "quandl":
        data = _quandl_request(names, start_date, end_date)
    # check pf_allocation:
    if pf_allocation is None:
        pf_allocation = _generate_pf_allocation(names=names)
    # build portfolio:
    pf = _build_portfolio_from_df(data, pf_allocation)
    return pf


def _stocknames_in_data_columns(names, df):
    """Returns True if at least one element of names was found as a column
    label in the dataframe df.
    """
    return any((name in label for name in names for label in df.columns))


def _generate_pf_allocation(names=None, data=None):
    """Takes column names of provided ``pandas.DataFrame`` ``data``, and generates a
    ``pandas.DataFrame`` with columns ``Name`` and ``Allocation`` which contain the
    names found in input ``data`` and 1.0/len(data.columns) respectively.

    :Input:
     :data: A ``pandas.DataFrame`` which contains prices of the stocks

    :Output:
     :pf_allocation: ``pandas.DataFrame`` with columns ``Name`` and ``Allocation``, which
         contain the names and weights of the stocks
    """
    # checking input arguments
    if names is not None and data is not None or names is None and data is None:
        raise ValueError("Pass one of the two: 'names' or 'data'.")
    if names is not None and not isinstance(names, list):
        raise ValueError("names is expected to be of type 'list'.")
    if data is not None and not isinstance(data, pd.DataFrame):
        raise ValueError("data is expected to be of type 'pandas.DataFrame'.")
    # if data is given:
    if data is not None:
        # this case is more complex, as we need to check for column labels in
        # data
        names = data.columns
        # sanity check: split names at '-' and take the leading part of the
        # split string, and check if this occurs in any of the other names.
        # if so, we treat this as a duplication, and ask the user to provide
        # a DataFrame with one data column per stock.
        splitnames = [name.split("-")[0].strip() for name in names]
        for i in range(len(splitnames)):
            splitname = splitnames[i]
            reducedlist = [elt for num, elt in enumerate(splitnames) if not num == i]
            if splitname in reducedlist:
                errormsg = (
                    "'data' pandas.DataFrame contains conflicting "
                    + "column labels."
                    + "\nMultiple columns with a substring of "
                    + "\n "
                    + str(splitname)
                    + "\n"
                    + "were found. You have two options:"
                    + "\n 1. call 'build_portfolio' and pass a "
                    + "pandas.DataFrame 'pf_allocation' that contains the "
                    + "weights/allocation of stocks within your "
                    + "portfolio. 'build_portfolio' will then extract "
                    + "the columns from 'data' that match the values "
                    + "of the column 'Name' in the pandas.DataFrame "
                    + "'pf_allocation'."
                    + "\n 2. call 'build_portfolio' and pass a "
                    + "pandas.DataFrame 'data' that does not have conflicting "
                    + "column labels, e.g. 'GOOG' and "
                    + "'GOOG - Adj. Close' are considered "
                    + "conflicting column headers."
                )
                raise ValueError(errormsg)
    # if names is given, we go directly to the below:
    # compute equal weights
    weights = [1.0 / len(names) for i in range(len(names))]
    return pd.DataFrame({"Allocation": weights, "Name": names})


def _build_portfolio_from_df(data, pf_allocation=None, datacolumns=["Adj. Close"]):
    """Returns a portfolio based on input in form of ``pandas.DataFrame``.

    :Input:
     :data: A ``pandas.DataFrame`` which contains prices of the stocks listed in
         pf_allocation
     :pf_allocation: (optional) ``pandas.DataFrame`` with the required data column
         labels ``Name`` and ``Allocation`` of the stocks. If not given, it is
         automatically generated with an equal weights for all stocks
         in the resulting portfolio.
     :datacolumns: (optional) A list of strings of data column labels
         to be extracted and returned (default: ["Adj. Close"]).

    :Output:
     :pf: Instance of Portfolio which contains all the information
         requested by the user.
    """
    # if pf_allocation is None, automatically generate it
    if pf_allocation is None:
        pf_allocation = _generate_pf_allocation(data=data)
    # make sure stock names are in data dataframe
    if not _stocknames_in_data_columns(pf_allocation.Name.values, data):
        raise ValueError(
            "Error: None of the provided stock names were"
            + "found in the provided dataframe."
        )
    # extract only "Adjusted Close" price ("Adj. Close" in quandl, "Adj Close" in yfinance)
    # column from DataFrame:
    data = _get_stocks_data_columns(data, pf_allocation.Name.values, datacolumns)
    # building portfolio:
    pf = Portfolio()
    for i in range(len(pf_allocation)):
        # get name of stock
        name = pf_allocation.loc[i].Name
        # extract data column(s) of said stock
        stock_data = data.loc[:, [name]].copy(deep=True)
        # if only one data column per stock exists, give dataframe a name
        if len(datacolumns) == 1:
            stock_data.name = datacolumns[0]
        # create Stock instance and add it to portfolio
        pf.add_stock(Stock(pf_allocation.loc[i], data=stock_data))
    return pf


def _all_list_ele_in_other(l1, l2):
    """Returns True if all elements of list l1 are found in list l2."""
    return all(ele in l2 for ele in l1)


def _any_list_ele_in_other(l1, l2):
    """Returns True if any element of list l1 is found in list l2."""
    return any(ele in l2 for ele in l1)


def _list_complement(A, B):
    """Returns the relative complement of A in B (also denoted as A\\B)"""
    return list(set(B) - set(A))


def build_portfolio(**kwargs):
    """This function builds and returns an instance of ``Portfolio``
    given a set of input arguments.

    :Input:
     :pf_allocation: (optional) ``pandas.DataFrame`` with the required data column
         labels ``Name`` and ``Allocation`` of the stocks. If not given, it is
         automatically generated with an equal weights for all stocks
         in the resulting portfolio.
     :names: (optional) A ``string`` or ``list`` of ``strings``, containing the names
         of the stocks, e.g. "GOOG" for Google.
     :start_date: (optional) ``string``/``datetime`` start date of stock data to be
         requested through `quandl`/`yfinance` (default: ``None``).
     :end_date: (optional) ``string``/``datetime`` end date of stock data to be
         requested through `quandl`/`yfinance` (default: ``None``).
     :data: (optional) A ``pandas.DataFrame`` which contains quantities of
         the stocks listed in ``pf_allocation``.
     :data_api: (optional) A ``string`` (default: ``quandl``) which determines how to
         obtain stock prices, if data is not provided by the user. Valid values:

         - ``quandl`` (Python package/API to `Quandl`)
         - ``yfinance`` (Python package formerly known as ``fix-yahoo-finance``)

    :Output:
     :pf: Instance of ``Portfolio`` which contains all the information
         requested by the user.

    .. note:: Only the following combinations of inputs are allowed:

     - ``names``, ``pf_allocation`` (optional), ``start_date`` (optional), ``end_date`` (optional), data_api (optional)
     - ``data``, ``pf_allocation`` (optional)

     The two different ways this function can be used are useful for:

     1. building a portfolio by pulling data from `quandl`/`yfinance`,
     2. building a portfolio by providing stock data which was obtained otherwise,
        e.g. from data files.

     If used in an unsupported way, the function (or subsequently called function) raises appropriate Exceptions
     with useful information what went wrong.
    """
    docstring_msg = (
        "Please read through the docstring, "
        "'build_portfolio.__doc__' and/or have a look at the "
        "examples in `examples/`."
    )
    input_error = (
        "You passed an unsupported argument to "
        "build_portfolio. The following arguments are not "
        "supported:"
        "\n {}\nOnly the following arguments are allowed:\n "
        "{}\n" + docstring_msg
    )
    input_comb_error = (
        "Error: None of the input arguments {} are allowed "
        "in combination with {}.\n" + docstring_msg
    )

    # list of all valid optional input arguments
    all_input_args = [
        "pf_allocation",
        "names",
        "start_date",
        "end_date",
        "data",
        "data_api",
    ]

    # check if no input argument was passed
    if kwargs == {}:
        raise ValueError(
            "Error:\nbuild_portfolio() requires input " + "arguments.\n" + docstring_msg
        )
    # check for valid input arguments
    if not _all_list_ele_in_other(kwargs.keys(), all_input_args):
        unsupported_input = _list_complement(all_input_args, kwargs.keys())
        raise ValueError(
            "Error:\n" + input_error.format(unsupported_input, all_input_args)
        )

    # create an empty portfolio
    pf = Portfolio()

    # 1. pf_allocation, names, start_date, end_date, data_api
    allowed_mandatory_args = ["names"]
    allowed_input_args = [
        "names",
        "pf_allocation",
        "start_date",
        "end_date",
        "data_api",
    ]
    complement_input_args = _list_complement(allowed_input_args, all_input_args)
    if _all_list_ele_in_other(allowed_mandatory_args, kwargs.keys()):
        # check that no input argument conflict arises:
        if _any_list_ele_in_other(complement_input_args, kwargs.keys()):
            raise ValueError(
                input_comb_error.format(complement_input_args, allowed_mandatory_args)
            )
        # get portfolio:
        pf = _build_portfolio_from_api(**kwargs)

    # 2. pf_allocation, data
    allowed_mandatory_args = ["data"]
    allowed_input_args = ["data", "pf_allocation"]
    complement_input_args = _list_complement(allowed_input_args, all_input_args)
    if _all_list_ele_in_other(allowed_mandatory_args, kwargs.keys()):
        # check that no input argument conflict arises:
        if _any_list_ele_in_other(complement_input_args, kwargs.keys()):
            raise ValueError(
                input_comb_error.format(complement_input_args, allowed_mandatory_args)
            )
        # get portfolio:
        pf = _build_portfolio_from_df(**kwargs)

    # final check
    if (
        pf.portfolio.empty
        or pf.data.empty
        or pf.stocks == {}
        or pf.expected_return is None
        or pf.volatility is None
        or pf.downside_risk is None
        or pf.sharpe is None
        or pf.skew is None
        or pf.kurtosis is None
    ):
        raise ValueError(
            "Should not get here. Something went wrong while "
            + "creating an instance of Portfolio."
            + docstring_msg
        )

    return pf
