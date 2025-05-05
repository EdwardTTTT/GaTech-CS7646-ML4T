""""""                   
"""MC2-P1: Market simulator.                   
                   
Copyright 2018, Georgia Institute of Technology (Georgia Tech)                   
Atlanta, Georgia 30332                   
All Rights Reserved                   
                   
Template code for CS 4646/7646                   
                   
Georgia Tech asserts copyright ownership of this template and all derivative                   
works, including solutions to the projects assigned in this course. Students                   
and other users of this template code are advised not to share it with others                   
or to make it available on publicly viewable websites including repositories                   
such as github and gitlab.  This copyright statement should not be removed                   
or edited.                   
                   
We do grant permission to share solutions privately with non-students such                   
as potential employers. However, sharing with other current or future                   
students of CS 7646 is prohibited and subject to being investigated as a                   
GT honor code violation.                   
                   
-----do not edit anything above this line---                   
                   
Student Name: Guangqing Tai                 
GT User ID: gtai3                  
GT ID: 903968079                

This code is a re-submission of Summer 2024 work, which may contain previous code
"""                   


import datetime as dt                   
import os                   
                   
import numpy as np                   
                   
import pandas as pd                   
from util import get_data, plot_data                   

def author():
    return 'gtai3' # replace tb34 with your Georgia Tech username. 
                   
def compute_portvals(
    orders_file="./orders/orders.csv",
    start_val=1000000,
    commission=9.95,
    impact=0.005,
):
    """
    Computes the portfolio values.
    
    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # Read the orders file
    print(f"Reading orders from {orders_file}")
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True)
    print("Orders:")
    print(orders.head())
    
    orders.sort_index(inplace=True)

    # Extract start and end dates from the orders
    start_date = orders.index.min()
    end_date = orders.index.max()
    print(f"Start date: {start_date}, End date: {end_date}")

    # Get the list of symbols
    symbols = orders['Symbol'].unique()
    print(f"Symbols: {symbols}")

    # Fetch historical stock data
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # Adjusted closing prices
    prices_all['Cash'] = 1.0  # Add a 'Cash' column with value 1.0 for easier calculations
    print("Prices All:")
    print(prices_all.head())

    # Initialize the trades DataFrame
    trades = pd.DataFrame(index=prices_all.index, columns=prices_all.columns)
    trades.fillna(0, inplace=True)
    print("Initialized Trades:")
    print(trades.head())

    # Process orders and update trades
    for date, order in orders.iterrows():
        symbol = order['Symbol']
        shares = order['Shares']
        if order['Order'] == 'BUY':
            trades.at[date, symbol] += shares
            price = prices_all.at[date, symbol] * (1 + impact)
            trades.at[date, 'Cash'] -= (price * shares + commission)
        elif order['Order'] == 'SELL':
            trades.at[date, symbol] -= shares
            price = prices_all.at[date, symbol] * (1 - impact)
            trades.at[date, 'Cash'] += (price * shares - commission)
    print("Trades after processing orders:")
    print(trades.head())

    # Initialize holdings DataFrame
    holdings = trades.cumsum()
    holdings['Cash'] += start_val
    print("Holdings after cumsum and initial cash:")
    print(holdings.head())

    # Calculate the portfolio value for each day
    portvals = (holdings * prices_all).sum(axis=1)
    print("Portfolio values:")
    print(portvals.head())

    # Create a single-column DataFrame for the portfolio values
    portvals = pd.DataFrame(portvals, columns=['Portfolio Value'])

    return portvals
                   
                   
def test_code():                   
    """                   
    Helper function to test code                   
    """                   
    # this is a helper function you can use to test your code                   
    # note that during autograding his function will not be called.                   
    # Define input parameters                   
                   
    of = "./orders/orders2.csv"                   
    sv = 1000000                   
                   
    # Process orders                   
    portvals = compute_portvals(orders_file=of, start_val=sv)                   
    if isinstance(portvals, pd.DataFrame):                   
        portvals = portvals[portvals.columns[0]]  # just get the first column                   
    else:                   
        "warning, code did not return a DataFrame"                   
                   
    # Get portfolio stats                   
    # Here we just fake the data. you should use your code from previous assignments.                   
    start_date = dt.datetime(2008, 1, 1)                   
    end_date = dt.datetime(2008, 6, 1)                   
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [                   
        0.2,                   
        0.01,                   
        0.02,                   
        1.5,                   
    ]                   
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [                   
        0.2,                   
        0.01,                   
        0.02,                   
        1.5,                   
    ]                   
                   
    # Compare portfolio against $SPX                   
    print(f"Date Range: {start_date} to {end_date}")                   
    print()                   
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")                   
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")                   
    print()                   
    print(f"Cumulative Return of Fund: {cum_ret}")                   
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")                   
    print()                   
    print(f"Standard Deviation of Fund: {std_daily_ret}")                   
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")                   
    print()                   
    print(f"Average Daily Return of Fund: {avg_daily_ret}")                   
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")                   
    print()                   
    print(f"Final Portfolio Value: {portvals[-1]}")                   
                   
                   
if __name__ == "__main__":                   
    test_code()                   
