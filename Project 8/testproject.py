#this code is a re-submission of previous term, which may contains similar codes

import datetime as dt
from experiment1 import run_experiment1
from experiment2 import run_experiment2


def main():
    # Define parameters
    symbol = "JPM"
    in_sample_start = dt.datetime(2008, 1, 1)
    in_sample_end = dt.datetime(2009, 12, 31)
    out_sample_start = dt.datetime(2010, 1, 1)
    out_sample_end = dt.datetime(2011, 12, 31)
    sv = 100000  # Starting value of portfolio
    impact_values = [0.0, 0.1, 0.2]

    # Run Experiment 1 (In-sample and Out-of-sample comparison for Manual and Strategy Learner)
    print("Running Experiment 1...")
    run_experiment1(symbol, in_sample_start, in_sample_end, out_sample_start, out_sample_end, sv)
    # Run Experiment 2 (Impact analysis for different values)
    print("Running Experiment 2...")
    run_experiment2(symbol, in_sample_start, in_sample_end, sv, impact_values)


if __name__ == "__main__":
    main()