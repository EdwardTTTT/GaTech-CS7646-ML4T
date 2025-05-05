#this code is a re-submission of previous term, which may contains similar codes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from StrategyLearner import StrategyLearner
from util import get_data, plot_data
import re


def sanitize_filename(title):
    """Sanitize the title to create a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', '_', title) + '.png'


def compute_metrics(trades, prices, initial_cash):
    """Compute metrics based on trades and benchmark data."""
    # Calculate portfolio values
    position = trades['Trades'].cumsum()  # Cumulative position
    cash = initial_cash - (trades['Trades'] * prices).cumsum()  # Cash after each trade
    portfolio_value = position * prices + cash  # Total portfolio value over time

    # Normalize portfolio values
    normalized_portfolio_value = portfolio_value / initial_cash

    # Calculate metrics
    total_return = normalized_portfolio_value.iloc[-1] - 1
    num_trades = trades['Trades'].abs().sum()  # Sum absolute trades for total trade count

    return total_return, num_trades, normalized_portfolio_value


def plot_comparison(df_benchmark, trades_dict, prices, title, sv):
    plt.figure(figsize=(12, 6))

    # Ensure prices and benchmark trades are aligned
    if prices.empty or df_benchmark.empty:
        print("Error: Empty prices or benchmark DataFrame.")
        return

    # Calculate normalized benchmark portfolio values
    position_benchmark = df_benchmark['Trades'].cumsum()
    cash_benchmark = sv - (df_benchmark['Trades'] * prices).cumsum()
    portfolio_value_benchmark = position_benchmark * prices + cash_benchmark
    portfolio_value_benchmark = portfolio_value_benchmark.dropna()

    if portfolio_value_benchmark.empty:
        print("Error: Benchmark portfolio value is empty after calculations.")
        return

    # Normalize the benchmark portfolio values
    normalized_benchmark = portfolio_value_benchmark / portfolio_value_benchmark.iloc[0]

    # Plot benchmark
    plt.plot(normalized_benchmark.index, normalized_benchmark, label='Benchmark', color='purple')

    # Plot each strategy learner for different impacts
    for impact, (trades, normalized_portfolio_value) in trades_dict.items():
        plt.plot(normalized_portfolio_value.index, normalized_portfolio_value,
                 label=f'Strategy Learner (Impact={impact})')

    # Labels and legend
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend(loc='best')

    # Save and show plot
    filename = sanitize_filename(title)
    plt.savefig(filename, format='png')
    plt.show()


def run_experiment2(symbol, sd, ed, sv, impact_values):
    # Define date range and create benchmark with only first-day buy and last-day sell
    dates = pd.date_range(sd, ed)
    df_benchmark = pd.DataFrame(index=dates, columns=['Trades'])
    df_benchmark['Trades'] = 0
    df_benchmark.iloc[0, 0] = 1000  # Buy on the first day
    df_benchmark.iloc[-1, 0] = -1000  # Sell on the last day

    # Fetch stock prices for the date range
    prices = get_data([symbol], dates)
    prices = prices[symbol].fillna(method="ffill")  # Forward-fill any missing prices

    # Dictionary to store trades and portfolio values for each impact value
    trades_dict = {}

    for impact in impact_values:
        # Initialize StrategyLearner with the specified impact
        learner = StrategyLearner(verbose=False, impact=impact, commission=0.0)
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)

        # Get trades and compute portfolio value for the given impact
        trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

        # Compute metrics and normalized portfolio value for plotting
        total_return, num_trades, normalized_portfolio_value = compute_metrics(trades, prices, sv)
        trades_dict[impact] = (trades, normalized_portfolio_value)

        print(f'Impact: {impact} | Total Return: {total_return:.2f} | Number of Trades: {num_trades}')

    # Plot results
    plot_comparison(df_benchmark, trades_dict, prices, "Impact Analysis of Strategy Learner",sv)

