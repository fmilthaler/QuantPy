import pandas as pd
import numpy as np
from finquant.returns import cumulative_returns, daily_returns, weighted_mean_daily_returns
from finquant.returns import daily_log_returns, historical_mean_return

def test_cumulative_returns():
    orig = [
        list(range(10)),
        [0, -0.025, -0.05, -0.075, -0.1, -0.125, -0.15, -0.175, -0.2, -0.225],
    ]
    l1 = range(1, 11)
    l2 = [40 - i for i in range(10)]
    d = {"1": l1, "2": l2}
    df = pd.DataFrame(d)
    ret = cumulative_returns(df)
    assert all(abs(ret["1"].values - orig[0]) <= 1e-15)
    assert all(abs(ret["2"].values - orig[1]) <= 1e-15)
    # with dividend of 0.2
    orig = [
        [0.2 + i for i in range(10)],
        [0.005, -0.02, -0.045, -0.07, -0.095, -0.12, -0.145, -0.17, -0.195, -0.22],
    ]
    ret = cumulative_returns(df, 0.2)
    assert all(abs(ret["1"].values - orig[0]) <= 1e-15)
    assert all(abs(ret["2"].values - orig[1]) <= 1e-15)


def test_daily_returns():
    orig = [[1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4], [1.0 / 9, 1.0 / 10, 1.0 / 11, 1.0 / 12]]
    l1 = range(1, 6)
    l2 = [10 * 0.2 + i * 0.25 for i in range(1, 6)]
    d = {"1": l1, "2": l2}
    df = pd.DataFrame(d)
    ret = daily_returns(df)
    assert all(abs(ret["1"].values - orig[0]) <= 1e-15)
    assert all(abs(ret["2"].values - orig[1]) <= 1e-15)

def test_weighted_daily_mean_returns():
    l1 = [ 1., 1.5, 2.25, 3.375 ]
    l2 = [ 1., 2., 4., 8. ]
    expected = [ 0.5 * 0.25 + 1 * 0.75 for i in range(len(l1)-1) ]
    weights = np.array([ 0.25, 0.75 ])
    d = { "1": l1, "2": l2 }
    df = pd.DataFrame(d)
    ret = weighted_mean_daily_returns(df, weights)
    assert all(abs(ret - expected) <= 1e-15)

    d = { "1": l1 }
    expected = [ 0.5 for i in range(len(l1)-1) ]
    df = pd.DataFrame(d)
    ret = weighted_mean_daily_returns(df, np.array([1]))
    assert all(abs(ret - expected) <= 1e-15)

def test_daily_log_returns():
    orig = [
        [
            0.6931471805599453,
            0.4054651081081644,
            0.28768207245178085,
            0.22314355131420976,
        ],
        [
            0.10536051565782635,
            0.09531017980432493,
            0.0870113769896297,
            0.08004270767353636,
        ],
    ]
    l1 = range(1, 6)
    l2 = [10 * 0.2 + i * 0.25 for i in range(1, 6)]
    d = {"1": l1, "2": l2}
    df = pd.DataFrame(d)
    ret = daily_log_returns(df)
    ret

    assert all(abs(ret["1"].values - orig[0]) <= 1e-15)
    assert all(abs(ret["2"].values - orig[1]) <= 1e-15)


def test_historical_mean_return():
    orig = [13.178779135809942, 3.8135072274034982]
    l1 = range(1, 101)
    l2 = [10 * 0.2 + i * 0.25 for i in range(21, 121)]
    d = {"1": l1, "2": l2}
    df = pd.DataFrame(d)
    ret = historical_mean_return(df, freq=252)
    assert abs(ret["1"] - orig[0]) <= 1e-15
    assert abs(ret["2"] - orig[1]) <= 1e-15
