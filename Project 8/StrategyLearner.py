#this code is a re-submission of previous term, which may contains similar codes

import pandas as pd
import numpy as np
import datetime as dt
from indicators import *
from util import get_data, plot_data

def choose_factor(features):
    return np.random.randint(features.shape[1])


def build_tree(np_data, leaf_size):
    # If the data is small enough or all labels are the same, return a leaf
    if np_data.shape[0] <= leaf_size or np.all(np_data[:, -1] == np_data[0, -1]):
        return np.asarray([["leaf", float(np.median(np_data[:, -1])), "NA", "NA"]])

    # Choose the best feature to split on
    factor = choose_factor(np_data[:, 0:np_data.shape[1] - 1])
    split_val = np.median(np_data[:, factor])

    # Handle cases where the split value results in no data on one side
    if np_data[np_data[:, factor] > split_val].shape[0] == 0:
        split_val = np.mean(np_data[:, factor])

    # Recursively build the left and right subtrees
    left_tree = build_tree(np_data[np_data[:, factor] <= split_val], leaf_size)
    right_tree = build_tree(np_data[np_data[:, factor] > split_val], leaf_size)

    # Create the root node
    root = np.array(["x" + str(factor), split_val, 1, left_tree.shape[0] + 1])
    return np.vstack((np.vstack((root, left_tree)), right_tree))


class RTLearner(object):
    """
    Random Tree Learner class.
    """

    def __init__(self, leaf_size=5, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):

        return "gtai3"

    def add_evidence(self, X, y):

        if len(y.shape) == 1:
            y = np.reshape(y, (X.shape[0], -1))

        np_arr = np.append(X, y, axis=1)
        self.tree = build_tree(np_arr, leaf_size=self.leaf_size)

    def query(self, points):

        y1 = np.zeros((points.shape[0],), dtype=float)
        for j in range(points.shape[0]):
            i = 0
            while i < self.tree.shape[0]:
                var = self.tree[i][0]
                if var != "leaf":
                    var_index = int(var.replace("x", ""))
                    if points[j][var_index] <= float(self.tree[i][1]):
                        jump_index = int(self.tree[i][2])
                    else:
                        jump_index = int(self.tree[i][3])
                else:
                    y1[j] = float(self.tree[i][1])
                    break
                i += jump_index
        return y1


class StrategyLearner:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.model = RTLearner(leaf_size=10)  # Instantiate the RT Learner

    def add_evidence(self, symbol, sd, ed, sv):
        # Fetch historical data
        dates = pd.date_range(start=sd, end=ed)
        df0 = get_data([symbol], dates)
        df = df0[symbol].fillna(0).replace(0, method='bfill')

        # Add indicators
        df_adx = get_data_adx([symbol], dates)
        df_adx['RSI'] = compute_RSI(df0, symbol)
        adx = compute_adx(df_adx)
        bollinger_bands = get_bollinger_bands(df)
        df_adx = df_adx.join(bollinger_bands)

        # Prepare features and target for training
        df_adx['Future Price'] = df_adx['Adj Close'].shift(-1)
        df_adx['Return'] = df_adx['Future Price'] - df_adx['Adj Close']
        df_adx['Target'] = np.where(df_adx['Return'] > 0, 1, 0)  # Binary classification target

        features = ['Adj Close', 'Lower Band', 'Upper Band', 'BB Value', 'RSI', 'ADX']
        X = df_adx[features].fillna(0)
        y = df_adx['Target'].dropna()

        # Train the model
        self.model.add_evidence(X.values, y.values)

    def testPolicy(self, symbol, sd, ed, sv):
        # Fetch historical data
        dates = pd.date_range(start=sd, end=ed)
        df0 = get_data([symbol], dates)
        df = df0[symbol].fillna(0).replace(0, method='bfill')

        # Add indicators
        df_adx = get_data_adx([symbol], dates)
        df_adx['RSI'] = compute_RSI(df0, symbol)
        adx = compute_adx(df_adx)
        bollinger_bands = get_bollinger_bands(df)
        df_adx = df_adx.join(bollinger_bands)

        # Prepare features for prediction
        features = ['Adj Close', 'Lower Band', 'Upper Band', 'BB Value', 'RSI', 'ADX']
        X = df_adx[features].fillna(0)

        # Make predictions
        predictions = self.model.query(X.values)

        # Initialize trades DataFrame
        trades = pd.DataFrame(0, index=df_adx.index, columns=['Trades'])
        position = 0  # Track current position: +1000 for long, -1000 for short

        # Generate trades based on predictions
        for i in range(1, len(predictions)):
            if predictions[i] == 1 and position == 0:
                trades.iloc[i] = 1000  # Buy 1000 shares
                position = 1000
            elif predictions[i] == 0 and position == 0:
                trades.iloc[i] = -1000  # Sell 1000 shares
                position = -1000
            elif predictions[i] == 0 and position == 1000:
                trades.iloc[i] = -1000  # Sell to close long position
                position = 0
            elif predictions[i] == 1 and position == -1000:
                trades.iloc[i] = 1000  # Buy to close short position
                position = 0

        return trades
