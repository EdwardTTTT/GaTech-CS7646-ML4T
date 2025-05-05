#This code is a re-submission of Summer 2024 work, which may contain previous code

import TheoreticallyOptimalStrategy as tos
import indicators as ind
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data

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
    bollinger_bands = ind.get_bollinger_bands(prices)

    # Compute RSI and plot
    rsi = ind.compute_RSI(prices_df, symbol)

    # Compute Momentum and plot
    momentum = ind.compute_momentum(prices)

    # Compute Fibonacci retracement levels and plot
    levels = ind.compute_fibonacci_retracement(prices)

    # Compute ADX and plot
    adx = ind.compute_adx(ind.get_data_adx([symbol], dates))

    # Plot and display statistics
    tos.plot_strategy(symbol, sd, ed, sv)