#This work is a resubmission of CS 7646 Summer 2024 for Fall 2024 term, which may contain partial of previous codes

import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return "gtai3"

    def add_evidence(self, data_x, data_y):
        data = np.column_stack((data_x, data_y))
        self.tree = self.build_tree(data)

    def build_tree(self, data):
        if data.shape[0] == 0:
            return np.array([[-1, np.nan, np.nan, np.nan]])

        if data.shape[0] <= self.leaf_size or np.all(data[:, -1] == data[0, -1]):
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        best_feature = np.random.randint(0, data.shape[1] - 1)
        split_val = np.median(data[:, best_feature])

        if np.all(data[:, best_feature] == split_val):
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        left_split = data[data[:, best_feature] <= split_val]
        right_split = data[data[:, best_feature] > split_val]

        if len(left_split) == 0 or len(right_split) == 0:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        left_tree = self.build_tree(left_split)
        right_tree = self.build_tree(right_split)

        root = np.array([[best_feature, split_val, 1, len(left_tree) + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        predictions = []
        for point in points:
            node = 0
            while self.tree[node, 0] != -1:
                feature = int(self.tree[node, 0])
                if point[feature] <= self.tree[node, 1]:
                    node += int(self.tree[node, 2])
                else:
                    node += int(self.tree[node, 3])
            predictions.append(self.tree[node, 1])
        return np.array(predictions)
