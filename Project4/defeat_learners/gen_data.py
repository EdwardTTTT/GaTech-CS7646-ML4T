""""""
"""  		  	   		 	   		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
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

Student Name: Tucker Balch (replace with your name)  		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: gtai3 (replace with your User ID)  		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 903968079 (replace with your GT ID)  		  	   		 	   		  		  		    	 		 		   		 		  
"""


import numpy as np

def best_4_lin_reg(seed=1489683272):
    """
    Returns data that performs significantly better with LinRegLearner than DTLearner.
    The data set should include from 2 to 10 columns in X, and one column in Y.
    The data should contain from 10 (minimum) to 1000 (maximum) rows.

    :param seed: The random seed for your data generation.
    :type seed: int
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.
    :rtype: numpy.ndarray
    """
    np.random.seed(seed)
    num_rows = np.random.randint(10, 1001)  # Randomly select the number of rows between 10 and 1000
    num_features = np.random.randint(2, 11)  # Randomly select the number of features between 2 and 10

    # Generate X matrix with uniform random values
    X = np.random.uniform(-10, 10, size=(num_rows, num_features))

    # Create a target Y that has a strong linear relationship with the features
    Y = 200 * np.sum(X, axis=1) -100

    return X, Y


# Function to generate data that favors decision tree learners
def best_4_dt(seed=1489683272,num_clusters=5, num_samples=999, spread=0.01):
    """
    Generates a dataset where DTLearner significantly outperforms LinRegLearner.
    The dataset contains between 10 and 1000 rows and 2 to 10 columns, based on the random seed provided.

    :param seed: Random seed to ensure reproducibility
    :type seed: int
    :return: Feature matrix X and target vector Y
    :rtype: tuple of numpy.ndarray
    """
    np.random.seed(seed)

    samples_per_cluster = num_samples // num_clusters  # Divide samples equally among clusters
    X = []
    y = []

    for cluster_idx in range(num_clusters):
        # Choose a random center for this cluster within a fixed range
        cluster_center = np.random.uniform(-10, 10, size=(2,))  # Center in 2D space

        # Generate points around this center using normal distribution
        points = np.random.normal(loc=cluster_center, scale=spread, size=(samples_per_cluster, 2))

        # Add the points and their corresponding cluster label
        X.append(points)
        y.append(np.full(samples_per_cluster, cluster_idx))

    # Combine all clusters into a single dataset
    X = np.vstack(X)
    y = np.concatenate(y)

    return X, y


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "gtai3"  # Change this to your user ID


if __name__ == "__main__":
    print("they call me Tim.")
