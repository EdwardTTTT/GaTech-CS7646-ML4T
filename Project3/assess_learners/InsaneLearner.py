#This work is a resubmission of CS 7646 Summer 2024 for Fall 2024 term, which may contain partial of previous codes

import numpy as np
import BagLearner as bl
import RTLearner as rt

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.bag_learners = [bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=20) for _ in range(20)]

    def author(self):
        return "gtai3"

    def add_evidence(self, data_x, data_y):
        for bag_learner in self.bag_learners:
            bag_learner.add_evidence(data_x, data_y)

    def query(self, points):
        predictions = np.array([bag_learner.query(points) for bag_learner in self.bag_learners])
        return np.mean(predictions, axis=0)
