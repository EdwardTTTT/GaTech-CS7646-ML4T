#this code is a re-submission of previous term, which may contains similar codes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import re
from util import get_data, plot_data

def sanitize_filename(title):
    """Sanitize the title to create a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', '_', title) + '.png'


def plot_comparison(symbol, df_benchmark, trades_manual, trades_learner, title,sv):
    # Fetch stock prices for the date range
    dates = trades_manual.index.union(trades_learner.index).union(df_benchmark.index)
    prices = get_data([symbol], dates)
    prices = prices[symbol].fillna(method="ffill")  # Forward-fill any missing prices

    # Calculate benchmark portfolio values
    position_benchmark = df_benchmark['Trades'].cumsum()
    cash_benchmark = sv - (df_benchmark['Trades'] * prices).cumsum()
    portfolio_value_benchmark = position_benchmark * prices + cash_benchmark
    portfolio_value_benchmark = portfolio_value_benchmark.dropna()
    normalized_benchmark = portfolio_value_benchmark / portfolio_value_benchmark.iloc[0]

    # Calculate manual strategy portfolio values
    position_manual = trades_manual['Trades'].cumsum()
    cash_manual = sv - (trades_manual['Trades'] * prices).cumsum()
    portfolio_value_manual = position_manual * prices + cash_manual
    normalized_manual = portfolio_value_manual / portfolio_value_manual.iloc[0]

    # Calculate strategy learner portfolio values
    position_learner = trades_learner['Trades'].cumsum()
    cash_learner = sv - (trades_learner['Trades'] * prices).cumsum()
    portfolio_value_learner = position_learner * prices + cash_learner
    normalized_learner = portfolio_value_learner / portfolio_value_learner.iloc[0]

    # Plot benchmark, manual strategy, and strategy learner
    plt.figure(figsize=(12, 6))
    plt.plot(normalized_benchmark.index, normalized_benchmark, label='Benchmark', color='purple')
    plt.plot(trades_manual.index, normalized_manual, label='Manual Strategy', color='red')
    plt.plot(trades_learner.index, normalized_learner, label='Strategy Learner', color='blue')

    # Labels and legend
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend(loc='best')
    filename = sanitize_filename(title)  # Sanitize the title to create a valid filename
    plt.savefig(filename, format='png')
    plt.show()


def run_experiment1(symbol, in_sample_start, in_sample_end, out_sample_start, out_sample_end, sv):
    # Initialize strategies
    ms = ManualStrategy()
    learner = StrategyLearner(verbose=False, impact=0.0, commission=0.0)

    # In-sample trading
    print("Running in-sample trading...")
    trades_manual_in = ms.testPolicy(symbol=symbol, sd=in_sample_start, ed=in_sample_end, sv=sv)

    # Create benchmark trades for in-sample
    dates_in = pd.date_range(in_sample_start, in_sample_end)
    df_benchmark_in = pd.DataFrame(index=dates_in, columns=['Trades'])
    df_benchmark_in['Trades'] = 0
    df_benchmark_in.iloc[0, 0] = 1000  # Buy on the first day
    df_benchmark_in.iloc[-1, 0] = -1000  # Sell on the last day

    # Compute portfolio values of StrategyLearner with in-sample data
    learner.add_evidence(symbol=symbol, sd=in_sample_start, ed=in_sample_end, sv=sv)
    trades_learner_in = learner.testPolicy(symbol=symbol, sd=in_sample_start, ed=in_sample_end, sv=sv)

    # Plot in-sample results
    plot_comparison(symbol,df_benchmark_in, trades_manual_in, trades_learner_in, "In-Sample Performance Comparison",sv)

    # Out-of-sample trading
    print("Running out-of-sample trading...")
    trades_manual_out = ms.testPolicy(symbol=symbol, sd=out_sample_start, ed=out_sample_end, sv=sv)

    # Create benchmark trades for out-of-sample
    dates_out = pd.date_range(out_sample_start, out_sample_end)
    df_benchmark_out = pd.DataFrame(index=dates_out, columns=['Trades'])
    df_benchmark_out['Trades'] = 0
    df_benchmark_out.iloc[0, 0] = 1000  # Buy on the first day
    df_benchmark_out.iloc[-1, 0] = -1000  # Sell on the last day

    # Compute portfolio values of StrategyLearner with out-of-sample data
    trades_learner_out = learner.testPolicy(symbol=symbol, sd=out_sample_start, ed=out_sample_end, sv=sv)

    # Plot out-of-sample results
    plot_comparison(symbol,df_benchmark_out, trades_manual_out, trades_learner_out, "Out-of-Sample Performance Comparison",sv)
