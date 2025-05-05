import math
import sys
import numpy as np
import time

import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bagl
import InsaneLearner as insl

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    inf = open(sys.argv[1])

    # Skip the header line and ignore the first column (date column)
    data = np.array([list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]])

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, :-1]  # All columns except the last one for training
    train_y = data[:train_rows, -1]   # The last column for the target variable
    test_x = data[train_rows:, :-1]   # All columns except the last one for testing
    test_y = data[train_rows:, -1]    # The last column for the target variable

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

    for name, learner in learners:
        print(f"\nTesting {name}")

        # Measure training time
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

        # Call the evaluate method to generate and save the RMSE comparison plot
        if hasattr(learner, "evaluate"):
            print(f"Generating RMSE comparison plot for {name}...")
            learner.evaluate(test_x, test_y)
            print(f"Plot saved for {name}")
