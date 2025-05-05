#this code is a re-submission of previous term, which may contains similar codes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import symbol_to_path

def author():
    return 'gtai3'


def get_bollinger_bands(prices, window=20):
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

    return bollinger_bands


def compute_RSI(df, symbol, window=14):
    df = df[[symbol]].rename(columns={symbol: 'Adj Close'})
    # Calculate daily returns
    delta = df['Adj Close'].diff()

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

    return rsi


def get_data_adx(symbols, dates):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        file_path = symbol_to_path(symbol)
        df_temp = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        df_temp = df_temp.rename(columns={'Adj Close': 'Adj Close'})
        df = df.join(df_temp, how='inner')
    df = df.dropna()
    df = df.fillna(method='bfill')
    return df


def compute_adx(prices, window=14):
    # Calculate True Range (TR), Positive Directional Movement (PDM), and Negative Directional Movement (NDM)
    prices['Previous Close'] = prices['Adj Close'].shift(1)
    prices['High-Low'] = prices['High'] - prices['Low']
    prices['High-Previous Close'] = np.abs(prices['High'] - prices['Previous Close'])
    prices['Low-Previous Close'] = np.abs(prices['Low'] - prices['Previous Close'])
    prices['TR'] = prices[['High-Low', 'High-Previous Close', 'Low-Previous Close']].max(axis=1)

    prices['Move Up'] = prices['High'] - prices['High'].shift(1)
    prices['Move Down'] = prices['Low'].shift(1) - prices['Low']

    prices['PDM'] = np.where((prices['Move Up'] > prices['Move Down']) & (prices['Move Up'] > 0), prices['Move Up'], 0)
    prices['NDM'] = np.where((prices['Move Down'] > prices['Move Up']) & (prices['Move Down'] > 0), prices['Move Down'],
                             0)

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
                 'Move Up', 'Move Down', 'TR', 'PDM', 'NDM', 'ATR', 'APDM', 'ANDM', 'PDI', 'NDI', 'DX'], axis=1,
                inplace=True)

    return prices[['ADX']]

