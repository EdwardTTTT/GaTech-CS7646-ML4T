#This code is a re-submission of Summer 2024 work, which may contain previous code

import pandas as pd
import util

def author():
    return 'gtai3'  # Replace with your GT username

def get_bollinger_bands(prices, window=20):
    """
    Compute Bollinger Bands for a given stock and plot it.

    :param prices: DataFrame containing stock prices with 'Close' column
    :type prices: pd.DataFrame
    :param window: The window size for calculating Bollinger Bands, default is 20
    :type window: int
    :return: A Pandas DataFrame containing the Bollinger Band values
    :rtype: pd.DataFrame
    """
    # Calculate the Simple Moving Average (SMA)
    sma = prices.rolling(window=window).mean()
    # Calculate the rolling standard deviation
    rolling_std = prices.rolling(window=window).std()
    # Calculate upper and lower bands
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    # Calculate Bollinger Band values
    bb_value = (prices - sma) / (2 * rolling_std)
    
    # Create a DataFrame to hold Bollinger Bands and BB value
    bollinger_bands = pd.DataFrame(index=prices.index)
    bollinger_bands['SMA'] = sma
    bollinger_bands['Upper Band'] = upper_band
    bollinger_bands['Lower Band'] = lower_band
    bollinger_bands['BB Value'] = bb_value

    # Plotting the Bollinger Bands and Closing Price
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Close Price', color='blue')
    plt.plot(sma, label='20-day SMA', color='orange')
    plt.plot(upper_band, label='Upper Band', color='red')
    plt.plot(lower_band, label='Lower Band', color='green')
    plt.fill_between(prices.index, lower_band.squeeze(), upper_band.squeeze(), color='grey', alpha=0.2)
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    # Save the plot as a PNG file
    plt.savefig("Bollinger_bands.png")

    return bollinger_bands

def compute_RSI(df, symbol, window=14):
    """
    Compute the Relative Strength Index (RSI) for a given stock and plot it.

    :param df: DataFrame containing stock prices with 'Close' column
    :type df: pd.DataFrame
    :param symbol: Stock symbol
    :type symbol: str
    :param window: The window size for calculating RSI, default is 14
    :type window: int
    :return: A Pandas Series containing the RSI values
    :rtype: pd.Series
    """
    df = df[[symbol]].rename(columns={symbol: 'Close'})
    # Calculate daily returns
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Plotting the RSI and Closing Price
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(df['Close'], label=f'{symbol} Close Price')
    ax1.set_title(f'{symbol} Closing Price')
    ax1.set_ylabel('Price')
    ax1.legend(loc='best')
    
    ax2.plot(rsi, label='RSI', color='orange')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig("RSI.png")
    
    return rsi

def compute_momentum(prices, window=20):
    """
    Compute the momentum for a given stock and plot it.

    :param prices: DataFrame containing stock prices with 'Close' column
    :type prices: pd.DataFrame
    :param window: The window size for calculating momentum, default is 20
    :type window: int
    :return: A Pandas DataFrame containing the momentum values
    :rtype: pd.DataFrame
    """
    # Calculate momentum
    momentum = (prices / prices.shift(window)) - 1
    
    # Create a DataFrame to hold momentum values
    momentum_df = pd.DataFrame(momentum, columns=['Momentum'])

    # Plotting the Closing Price and Momentum
    plt.figure(figsize=(12, 6))
    
    # Plot closing price
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Close Price', color='blue')
    plt.title('Stock Prices and Momentum')
    plt.ylabel('Price')
    plt.legend(loc='best')
    
    # Plot momentum
    plt.subplot(2, 1, 2)
    plt.plot(momentum, label='Momentum', color='red')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Horizontal line at 0
    plt.xlabel('Date')
    plt.ylabel('Momentum')
    plt.legend(loc='best')
    
    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig("momentum.png")

    return momentum_df

def compute_fibonacci_retracement(prices):
    """
    Compute the Fibonacci retracement levels for a given stock and plot them.

    :param prices: DataFrame containing stock prices with 'Close' column
    :type prices: pd.DataFrame
    :return: A dictionary containing the Fibonacci retracement levels
    :rtype: dict
    """
    max_price = prices.max()
    min_price = prices.min()

    diff = max_price - min_price

    levels = {
        '100%': max_price,
        '61.8%': max_price - 0.618 * diff,
        '50%': max_price - 0.5 * diff,
        '38.2%': max_price - 0.382 * diff,
        '23.6%': max_price - 0.236 * diff,
        '0%': min_price
    }

    # Plotting the Closing Price and Fibonacci retracement levels
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Close Price', color='blue')
    
    for level, price in levels.items():
        plt.axhline(price, linestyle='--', alpha=0.5, label=f'Fibonacci {level} ({price:.2f})')

    plt.title('Fibonacci Retracement Levels')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    
    # Save the plot as a PNG file
    plt.savefig("fibonacci_retracement.png")

    return levels

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data_adx(symbols, dates):
    """Read stock data (all columns) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv(f"data/{symbol}.csv", index_col="Date", parse_dates=True)
        df_temp = df_temp.rename(columns={'Adj Close':'Adj Close'})
        df = df.join(df_temp, how='inner')
    df = df.dropna()
    df = df.fillna(method='bfill')
    return df

def compute_adx(prices, window=14):
    """
    Compute Average Directional Index (ADX) for a given stock and plot it.

    :param prices: DataFrame containing stock prices with all columns
    :type prices: pd.DataFrame
    :param window: The window size for calculating ADX, default is 14
    :type window: int
    :return: A Pandas DataFrame containing the ADX values
    :rtype: pd.DataFrame
    """
    # Calculate True Range (TR), Positive Directional Movement (PDM), and Negative Directional Movement (NDM)
    prices['Previous Close'] = prices['Close'].shift(1)
    prices['High-Low'] = prices['High'] - prices['Low']
    prices['High-Previous Close'] = np.abs(prices['High'] - prices['Previous Close'])
    prices['Low-Previous Close'] = np.abs(prices['Low'] - prices['Previous Close'])
    prices['TR'] = prices[['High-Low', 'High-Previous Close', 'Low-Previous Close']].max(axis=1)
    
    prices['Move Up'] = prices['High'] - prices['High'].shift(1)
    prices['Move Down'] = prices['Low'].shift(1) - prices['Low']
    
    prices['PDM'] = np.where((prices['Move Up'] > prices['Move Down']) & (prices['Move Up'] > 0), prices['Move Up'], 0)
    prices['NDM'] = np.where((prices['Move Down'] > prices['Move Up']) & (prices['Move Down'] > 0), prices['Move Down'], 0)
    
    # Calculate smoothed versions of TR, PDM, and NDM
    prices['ATR'] = prices['TR'].rolling(window=window).mean()
    prices['APDM'] = prices['PDM'].rolling(window=window).mean()
    prices['ANDM'] = prices['NDM'].rolling(window=window).mean()
    
    # Calculate Directional Movement Index (DX)
    prices['PDI'] = (prices['APDM'] / prices['ATR']) * 100
    prices['NDI'] = (prices['ANDM'] / prices['ATR']) * 100
    prices['DX'] = np.abs(prices['PDI'] - prices['NDI']) / (prices['PDI'] + prices['NDI']) * 100
    
    # Calculate Average Directional Index (ADX)
    prices['ADX'] = prices['DX'].rolling(window=window).mean()
    
    # Drop intermediate columns
    prices.drop(['Previous Close', 'High-Low', 'High-Previous Close', 'Low-Previous Close',
                 'Move Up', 'Move Down', 'TR', 'PDM', 'NDM', 'ATR', 'APDM', 'ANDM', 'PDI', 'NDI', 'DX'], axis=1, inplace=True)
    
    # Plotting the ADX and Adj Close Price
    plt.figure(figsize=(12, 8))
    
    # Plot Adj Close
    plt.subplot(2, 1, 1)
    plt.plot(prices['Adj Close'], label='Adj Close', color='blue')
    plt.title('Stock Prices and Average Directional Index (ADX)')
    plt.ylabel('Price')
    plt.legend(loc='best')
    
    # Plot ADX
    plt.subplot(2, 1, 2)
    plt.plot(prices['ADX'], label='ADX', color='purple')
    plt.axhline(y=20, color='r', linestyle='--', label='ADXR > 20')
    plt.axhline(y=40, color='g', linestyle='--', label='ADXR > 40')
    plt.axhline(y=60, color='b', linestyle='--', label='ADXR > 60')
    plt.axhline(y=80, color='y', linestyle='--', label='ADXR > 80')
    plt.xlabel('Date')
    plt.ylabel('ADX')
    plt.legend(loc='best')
    
    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig("ADX.png")

    return prices[['ADX']]

