"""MC1-P2: Optimize a portfolio.

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

This work is a resubmission of CS 7646 Summer 2024 for Fall 2024 term, which may contain partial of previous codes
"""
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sp
from util import get_data, plot_data

# Function to optimize portfolio
def optimize_portfolio(
    sd=dt.datetime(2008, 6, 1),
    ed=dt.datetime(2009, 6, 1),
    syms=["IBM", "X", "GLD", "JPM"],
    gen_plot=True,
):
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices = prices_all[syms]
    prices_SPY = prices_all["SPY"]

    num_stocks = len(syms)
    initial_guess = [1.0 / num_stocks] * num_stocks
    bounds = [(0.0, 1.0)] * num_stocks
    constraints = {'type': 'eq', 'fun': lambda allocs: 1.0 - np.sum(allocs)}

    def min_sharpe_ratio(allocs, prices):
        return -assess_portfolio(allocs, prices)[4]

    result = sp.minimize(
        min_sharpe_ratio, initial_guess, args=prices,
        method='SLSQP', bounds=bounds, constraints=constraints
    )
    allocs = result.x

    port_val, cr, adr, sddr, sr = assess_portfolio(allocs, prices)

    if gen_plot:
        normed_port_val = port_val / port_val.iloc[0]
        normed_SPY = prices_SPY / prices_SPY.iloc[0]
        df_temp = pd.concat([normed_port_val, normed_SPY], axis=1)
        df_temp.columns = ['Portfolio', 'SPY']

        ax = df_temp.plot(title="Daily portfolio value and SPY", fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized price')
        plt.savefig('Figure1.png')

    return allocs, cr, adr, sddr, sr

# Function to assess portfolio
def assess_portfolio(allocs, prices):
    normed = prices / prices.iloc[0]
    alloced = normed * allocs
    port_val = alloced.sum(axis=1)

    daily_rets = port_val.pct_change().dropna()

    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = (adr / sddr) * np.sqrt(252)

    return port_val, cr, adr, sddr, sr

# Function to test the code
def test_code():
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations: {allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")

if __name__ == "__main__":
    test_code()
