import TheoreticallyOptimalStrategy as tos
import indicators as ind
from util import get_data
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def author():
    return 'gtai3'  # Replace with your GT username


if __name__ == "__main__":
    # Load data
    symbol = 'JPM'
    dates = pd.date_range('2008-01-01', '2009-12-31')
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    prices_df = get_data([symbol], dates)
    prices = prices_df[symbol].fillna(0).replace(0, method='bfill')
    
    # Compute Bollinger Bands and plot
    bollinger_bands = indicators.get_bollinger_bands(prices)

    # Compute RSI and plot
    rsi = indicators.compute_RSI(prices_df, symbol)

    # Compute Momentum and plot
    momentum = indicators.compute_momentum(prices)

    # Compute Fibonacci retracement levels and plot
    levels = indicators.compute_fibonacci_retracement(prices)

    # Compute ADX and plot
    adx = indicators.compute_adx(indicators.get_data_adx([symbol], dates))

    # Plot and display statistics
    indicators.plot_strategy(symbol, sd, ed, sv)