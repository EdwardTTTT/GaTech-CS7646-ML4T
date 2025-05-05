#This work is a resubmission of CS 7646 Summer 2024 for Fall 2024 term, which may contain partial of previous codes

import math
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bagl
import InsaneLearner as insl


def calculate_mae(true_y, pred_y):
    return np.mean(np.abs(true_y - pred_y))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    inf = open(sys.argv[1])

    # Skip the header line and ignore the first column (date column)
    data = np.array([list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]])

    # Compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # Separate out training and testing data
    train_x = data[:train_rows, :-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, :-1]
    test_y = data[train_rows:, -1]

    print(f"Training Data: {train_x.shape}, {train_y.shape}")
    print(f"Testing Data: {test_x.shape}, {test_y.shape}")

    # List of learners to be tested
    learners = [
        ("LinRegLearner", lrl.LinRegLearner(verbose=True)),
        ("DTLearner", dtl.DTLearner(leaf_size=5, verbose=True)),
        ("RTLearner", rtl.RTLearner(leaf_size=5, verbose=True)),
        ("BagLearner", bagl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": 5}, bags=20, verbose=True)),
        ("InsaneLearner", insl.InsaneLearner(verbose=True)),
    ]

    # Metrics storage
    metrics = {}

    for name, learner in learners:
        print(f"\nTesting {name}")

        start_time = time.time()
        learner.add_evidence(train_x, train_y)
        training_time = time.time() - start_time
        print(f"Training time for {name}: {training_time:.4f} seconds")

        # In-sample evaluation
        start_time = time.time()
        pred_y_in = learner.query(train_x)
        query_time_in_sample = time.time() - start_time
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        c_in = np.corrcoef(pred_y_in, y=train_y)[0, 1]
        print(f"In-sample results for {name}:")
        print(f"  RMSE: {rmse_in}")
        print(f"  Correlation: {c_in}")
        print(f"  Query time for in-sample data: {query_time_in_sample:.4f} seconds")

        # Out-of-sample evaluation
        start_time = time.time()
        pred_y_out = learner.query(test_x)
        query_time_out_sample = time.time() - start_time
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        c_out = np.corrcoef(pred_y_out, y=test_y)[0, 1]
        print(f"Out-of-sample results for {name}:")
        print(f"  RMSE: {rmse_out}")
        print(f"  Correlation: {c_out}")
        print(f"  Query time for out-of-sample data: {query_time_out_sample:.4f} seconds")

        # Store the metrics
        metrics[name] = {
            "training_time": training_time,
            "rmse_in": rmse_in,
            "c_in": c_in,
            "query_time_in": query_time_in_sample,
            "rmse_out": rmse_out,
            "c_out": c_out,
            "query_time_out": query_time_out_sample
        }

    # Plot RMSE vs Leaf Size for DTLearner and BagLearner
    leaf_sizes = range(1, 51, 5)
    dt_rmse_in_sample = []
    dt_rmse_out_sample = []
    bag_rmse_in_sample = []
    bag_rmse_out_sample = []

    for leaf_size in leaf_sizes:
        # DTLearner evaluation
        dt_learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
        dt_learner.add_evidence(train_x, train_y)

        pred_y_in = dt_learner.query(train_x)
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        dt_rmse_in_sample.append(rmse_in)

        pred_y_out = dt_learner.query(test_x)
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        dt_rmse_out_sample.append(rmse_out)

        # BagLearner evaluation
        bag_learner = bagl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, verbose=False)
        bag_learner.add_evidence(train_x, train_y)

        pred_y_in_bag = bag_learner.query(train_x)
        rmse_in_bag = math.sqrt(((train_y - pred_y_in_bag) ** 2).sum() / train_y.shape[0])
        bag_rmse_in_sample.append(rmse_in_bag)

        pred_y_out_bag = bag_learner.query(test_x)
        rmse_out_bag = math.sqrt(((test_y - pred_y_out_bag) ** 2).sum() / test_y.shape[0])
        bag_rmse_out_sample.append(rmse_out_bag)

    # Plot RMSE vs Leaf Size for DTLearner
    plt.figure()
    plt.plot(leaf_sizes, dt_rmse_in_sample, label='In-sample RMSE', color='blue')
    plt.plot(leaf_sizes, dt_rmse_out_sample, label='Out-of-sample RMSE', color='orange')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('DTLearner RMSE vs Leaf Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig('DTLearner_RMSE_vs_LeafSize.png')
    plt.close()

    # Plot RMSE vs Leaf Size for BagLearner
    plt.figure()
    plt.plot(leaf_sizes, bag_rmse_in_sample, label='In-sample RMSE', color='blue')
    plt.plot(leaf_sizes, bag_rmse_out_sample, label='Out-of-sample RMSE', color='orange')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('BagLearner RMSE vs Leaf Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig('BagLearner_RMSE_vs_LeafSize.png')
    plt.close()

    # Compare DTLearner and RTLearner using MAE over multiple trials
    trials = 20
    dt_mae_trials = []
    rt_mae_trials = []

    for trial in range(trials):
        # Train and evaluate DTLearner
        dt_learner = dtl.DTLearner(leaf_size=5, verbose=False)
        dt_learner.add_evidence(train_x, train_y)
        dt_pred_y = dt_learner.query(test_x)
        dt_mae = calculate_mae(test_y, dt_pred_y)
        dt_mae_trials.append(dt_mae)

        # Train and evaluate RTLearner
        rt_learner = rtl.RTLearner(leaf_size=5, verbose=False)
        rt_learner.add_evidence(train_x, train_y)
        rt_pred_y = rt_learner.query(test_x)
        rt_mae = calculate_mae(test_y, rt_pred_y)
        rt_mae_trials.append(rt_mae)

    # Plot MAE vs Trials for both learners
    plt.figure()
    plt.plot(range(1, trials + 1), dt_mae_trials, label='DTLearner MAE', color='blue', marker='o')
    plt.plot(range(1, trials + 1), rt_mae_trials, label='RTLearner MAE', color='orange', marker='x')
    plt.xlabel('Trial Number')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('DTLearner vs RTLearner: MAE over Trials')
    plt.legend()
    plt.tight_layout()
    plt.savefig('DT_vs_RT_MAE_over_Trials.png')
    plt.close()

    print("\nPlots generated:")
    print("1. DTLearner_RMSE_vs_LeafSize.png")
    print("2. BagLearner_RMSE_vs_LeafSize.png")
    print("3. DT_vs_RT_MAE_over_Trials.png")
