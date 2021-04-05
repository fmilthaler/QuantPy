"""The module provides functions to compute quantities relevant to financial
portfolios, e.g. a weighted average, which is the expected value/return, a
weighted standard deviation (volatility), and the Sharpe ratio.
"""


import numpy as np
import pandas as pd
from finquant.returns import weighted_mean_daily_returns


def weighted_mean(means, weights):
    """Computes the weighted mean/average, or in the case of a
    financial portfolio, it can be used for the Expected Return
    of said portfolio.

    :Input:
     :means: ``numpy.ndarray``/``pd.Series`` of mean/average values
     :weights: ``numpy.ndarray``/``pd.Series`` of weights

    :Output:
     :weighted mu: ``numpy.ndarray``: ``(np.sum(means*weights))``
    """
    if not isinstance(weights, (pd.Series, np.ndarray)):
        raise ValueError("weights is expected to be a pandas.Series/np.ndarray")
    if not isinstance(means, (pd.Series, np.ndarray)):
        raise ValueError("means is expected to be a pandas.Series/np.ndarray")
    return np.sum(means * weights)


def weighted_std(cov_matrix, weights):
    """Computes the weighted standard deviation, or Volatility of
    a portfolio, which contains several stocks.

    :Input:
     :cov_matrix: ``numpy.ndarray``/``pandas.DataFrame``, covariance matrix
     :weights: ``numpy.ndarray``/``pd.Series`` of weights

    :Output:
     :weighted sigma: ``numpy.ndarray``:
         ``np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))``
    """
    if not isinstance(weights, (pd.Series, np.ndarray)):
        raise ValueError("weights is expected to be a pandas.Series, np.array")
    if not isinstance(cov_matrix, (np.ndarray, (np.ndarray, pd.DataFrame))):
        raise ValueError(
            "cov_matrix is expected to be a numpy.ndarray/pandas.DataFrame"
        )
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def sharpe_ratio(exp_return, volatility, risk_free_rate=0.005):
    """Computes the Sharpe Ratio

    :Input:
     :exp_return: ``int``/``float``, Expected Return of a portfolio
     :volatility: ``int``/``float``, Volatility of a portfolio
     :risk_free_rate: ``int``/``float`` (default= ``0.005``), risk free rate

    :Output:
     :sharpe ratio: ``float`` ``(exp_return - risk_free_rate)/float(volatility)``
    """
    if not isinstance(
        exp_return, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("exp_return is expected to be an integer or float.")
    if not isinstance(
        volatility, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("volatility is expected to be an integer or float.")
    if not isinstance(
        risk_free_rate, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("risk_free_rate is expected to be an integer or float.")
    return (exp_return - risk_free_rate) / float(volatility)

def sortino_ratio(exp_return, downside_risk, risk_free_rate=0.005):
    """Computes the Sharpe Ratio

    :Input:
     :exp_return: ``int``/``float``, Expected Return of a portfolio
     :downside_risk: ``int``/``float``/``NaN``, Downside Risk of a portfolio
     :risk_free_rate: ``int``/``float`` (default= ``0.005``), risk free rate

    :Output:
     :sharpe ratio: ``float`` ``(exp_return - risk_free_rate)/float(downside_risk)``
     Can be ``NaN`` if ``downside_risk`` is NaN
    """
    if not isinstance(
        exp_return, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("exp_return is expected to be an integer or float.")
    if not isinstance(
        downside_risk, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("volatility is expected to be an integer or float.")
    if not isinstance(
        risk_free_rate, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("risk_free_rate is expected to be an integer or float.")
    return (exp_return - risk_free_rate) / float(downside_risk)

def downside_risk(data, weights, risk_free_rate=0.005):
    """Computes the downside risk (target semideviation of returns, given a risk free rate)
    :Input:
      :data: ``pandas.DataFrame`` with daily stock prices
      :weights: ``numpy.ndarray``/``pd.Series`` of weights
      :risk_free_rate: ``int``/``float`` (default=``0.005``), risk free rate
    
    :Output:
      :downside_risk: ``float`` (can be NaN if all returns outperform the risk free rate)
    """
    wtd_daily_mean = weighted_mean_daily_returns(data, weights)
    under = np.where(wtd_daily_mean < risk_free_rate, wtd_daily_mean, np.NaN)
    under = under[np.logical_not(np.isnan(under))]
    if len(under) == 0:
        return np.NaN
    downside = under - risk_free_rate
    return np.sqrt((downside * downside).mean())



def annualised_portfolio_quantities(
    weights, means, cov_matrix, risk_free_rate=0.005, freq=252
):
    """Computes and returns the expected annualised return, volatility
    and Sharpe Ratio of a portfolio.

    :Input:
     :weights: ``numpy.ndarray``/``pd.Series`` of weights
     :means: ``numpy.ndarray``/``pd.Series`` of mean/average values
     :cov_matrix: ``numpy.ndarray``/``pandas.DataFrame``, covariance matrix
     :risk_free_rate: ``float`` (default= ``0.005``), risk free rate
     :freq: ``int`` (default= ``252``), number of trading days, default
         value corresponds to trading days in a year

    :Output:
     :(Expected Return, Volatility, Sharpe Ratio): tuple of those
         three quantities
    """
    if not isinstance(freq, int):
        raise ValueError("freq is expected to be an integer.")
    expected_return = weighted_mean(means, weights) * freq
    volatility = weighted_std(cov_matrix, weights) * np.sqrt(freq)
    sharpe = sharpe_ratio(expected_return, volatility, risk_free_rate)
    return (expected_return, volatility, sharpe)
