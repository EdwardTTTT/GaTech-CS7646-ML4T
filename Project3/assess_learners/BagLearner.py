#This work is a resubmission of CS 7646 Summer 2024 for Fall 2024 term, which may contain partial of previous codes

import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learners = [learner(**kwargs) for _ in range(bags)]
        self.bags = bags
        self.verbose = verbose

    def author(self):
        return "gtai3"

    def add_evidence(self, data_x, data_y):
        n = data_x.shape[0]
        for learner in self.learners:
            indices = np.random.choice(n, n, replace=True)
            learner.add_evidence(data_x[indices], data_y[indices])

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=0)
