import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from indicators import *
from util import get_data, plot_data


# Manual strategy
class ManualStrategy:
    def __init__(self):
        pass

    def testPolicy(self, symbol, sd, ed, sv):
        # Fetch historical data
        dates = pd.date_range(start=sd, end=ed)
        df0 = get_data([symbol], dates)
        df = df0[symbol].fillna(0).replace(0, method='bfill')

        # Add indicators
        df_adx = get_data_adx([symbol], dates)
        df_adx['RSI'] = compute_RSI(df0, symbol)
        df_adx['ADX'] = compute_adx(df_adx)['ADX']
        bollinger_bands = get_bollinger_bands(df)
        df_adx = df_adx.join(bollinger_bands)

        # Initialize strategy parameters
        position = 0  # Track current position: +1000 for long, -1000 for short
        trades = pd.DataFrame(0, index=df_adx.index, columns=["Trades"])

        # Generate signals based on combined indicators
        for index, row in df_adx.iterrows():
            price = row['Adj Close']
            lower_band = row['Lower Band']
            upper_band = row['Upper Band']
            rsi = row['RSI']
            adx = row['ADX']

            # Long signal
            if position == 0 and price <= lower_band and rsi < 30 and adx > 20:
                trades.loc[index] = 1000  # Buy 1000 shares
                position = 1000

            # Short signal
            elif position == 0 and price >= upper_band and rsi > 70 and adx > 20:
                trades.loc[index] = -1000  # Sell 1000 shares
                position = -1000

            # Closing long position
            elif position == 1000 and (price >= upper_band or rsi > 70 or adx < 20):
                trades.loc[index] = -1000  # Sell to exit long
                position = 0

            # Closing short position
            elif position == -1000 and (price <= lower_band or rsi < 30 or adx < 20):
                trades.loc[index] = 1000  # Buy to cover short
                position = 0

        return trades


def plot_performance(df_benchmark, trades, symbol, sv, title):
    # Get stock prices to calculate portfolio values
    prices = get_data([symbol], trades.index)
    prices = prices[symbol].fillna(method="ffill")

    # Calculate portfolio values
    position_manual = trades['Trades'].cumsum()
    cash_manual = sv - (trades['Trades'] * prices).cumsum()
    portfolio_value_manual = position_manual * prices + cash_manual
    portfolio_value_manual /= portfolio_value_manual.iloc[0]  # Normalize

    # Calculate benchmark portfolio
    position_benchmark = df_benchmark['Trades'].cumsum()
    cash_benchmark = sv - (df_benchmark['Trades'] * prices).cumsum()
    portfolio_value_benchmark = position_benchmark * prices + cash_benchmark
    portfolio_value_benchmark /= portfolio_value_benchmark.iloc[0]  # Normalize

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value_benchmark.index, portfolio_value_benchmark, label='Benchmark', color='purple')
    plt.plot(trades.index, portfolio_value_manual, label='Manual Strategy', color='red')

    # Add entry/exit points
    long_entries = trades[trades['Trades'] == 1000].index
    short_entries = trades[trades['Trades'] == -1000].index
    for date in long_entries:
        plt.axvline(x=date, color='blue', linestyle='--', label='Long Entry')
    for date in short_entries:
        plt.axvline(x=date, color='black', linestyle='--', label='Short Entry')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend(loc='best')
    plt.show()
