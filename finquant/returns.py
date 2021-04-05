"""The module provides functions to compute different kinds of returns of stocks."""


import numpy as np
import pandas as pd


def cumulative_returns(data, dividend=0):
    """Returns DataFrame with cumulative returns

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_0} + \\text{dividend}}{\\text{price}_{t_0}}`

    :Input:
     :data: ``pandas.DataFrame`` with daily stock prices
     :dividend: ``float`` (default= ``0``), paid dividend

    :Output:
     :ret: a ``pandas.DataFrame`` of cumulative Returns of given stock prices.
    """
    return data.dropna(axis=0, how="any").apply(lambda x: (x - x[0] + dividend) / x[0])


def daily_returns(data):
    """Returns DataFrame with daily returns (percentage change)

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}{\\text{price}_{t_{i-1}}}`

    :Input:
     :data: ``pandas.DataFrame`` with daily stock prices

    :Output:
     :ret: a ``pandas.DataFrame`` of daily percentage change of Returns
         of given stock prices.
    """
    return data.pct_change().dropna(how="all").replace([np.inf, -np.inf], np.nan)

def weighted_mean_daily_returns(data, weights):
    """Returns DataFrame with the daily weighted mean returns

    :Input:
      :data: ``pandas.DataFrame`` with daily stock prices
      :weights: ``numpy.ndarray``/``pd.Series`` of weights

    :Output:
      :ret: ``numpy.array`` of weighted mean daily percentage change of Returns
    """
    return np.dot(daily_returns(data), weights)

def daily_log_returns(data):
    """
    Returns DataFrame with daily log returns

    :math:`R_{\\log} = \\log\\left(1 + \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}{\\text{price}_{t_{i-1}}}\\right)`

    :Input:
     :data: ``pandas.DataFrame`` with daily stock prices

    :Output:
     :ret: a ``pandas.DataFrame`` of
         log(1 + daily percentage change of Returns)
    """
    return np.log(1 + daily_returns(data)).dropna(how="all")


def historical_mean_return(data, freq=252):
    """Returns the mean return based on historical stock price data.

    :Input:
     :data: ``pandas.DataFrame`` with daily stock prices
     :freq: ``int`` (default= ``252``), number of trading days, default
             value corresponds to trading days in a year

    :Output:
     :ret: a ``pandas.DataFrame`` of historical mean Returns.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas.DataFrame")
    return daily_returns(data).mean() * freq
