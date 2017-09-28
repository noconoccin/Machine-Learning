# CSC713M - Nocon, Nicco Louis

import numpy as np
from collections import Counter as ctr

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """
        For k-nearest neighbors training is just memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y
    
    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """

        # pass INSERT CODE HERE
		
        i, j = 0, 0 # Assign counters for dist array 
        M = X.shape[0] # Retrieves num_test for M (test) dimension
        N = self.X_train.shape[0] # Retrieves num_train for N (train) dimension
        dists = np.zeros((M, N)) # Initialize an MxN (500, 5000) array with values of 0
		
		# for i in m -- 'int' object is not iterable [SOLVED: range() generates list of int]
        for i in range(M):
            for j in range(N):
                # Euclidean Distance Formula: sqrt((q1 - p1)^2 + (p2 - p2)^2 + ... + (qn - pn)^2)
                diff_squared = np.square(X[i] - self.X_train[j])
                sum = np.sum(diff_squared)
                dists[i, j] = np.sqrt(sum) 
        return dists
    
    
    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Hint: Look up the function numpy.argsort.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        # pass INSERT CODE HERE
		
        num_test = dists.shape[0] # Retrieve num_test (500,)
        y = np.zeros(num_test)
        # print(y.shape) # Check num_test value (500,)
        
        for i in range(num_test): # per test example
            labels = [] # Initialize where to put labels
            labels = np.array(self.y_train[np.argsort(dists[i],axis=-1,kind='quicksort')[:k]])
            label_freq = np.bincount(labels) # Counts frequency per label 	
            y[i] = np.argmax(label_freq) # Returns highest value (most frequent label)
        return y
    
    def compute_distances_one_loop(self, X):
        # Same with compute_distances_two_loops
        # pass INSERT CODE HERE
	
        i, j = 0, 0 # Assign counters for dist array 
        M = X.shape[0] # Retrieves num_test for M (test) dimension
        N = self.X_train.shape[0] # Retrieves num_train for N (train) dimension
        dists = np.zeros((M, N)) # Initialize an MxN (500, 5000) array with values of 0
		
        for i in range(M):
            # Euclidean Distance Formula: sqrt((q1 - p1)^2 + (p2 - p2)^2 + ... + (qn - pn)^2)
            diff_squared = np.square(X[i] - self.X_train)
            sum = np.sum(diff_squared, axis=1)
            dists[i] = np.sqrt(sum) 
        return dists


    def compute_distances_no_loops(self, X):
        # Same with compute_distances_two_loops
        # pass INSERT CODE HERE
		
        M = X.shape[0] # Retrieves num_test for M (test) dimension
        N = self.X_train.shape[0] # Retrieves num_train for N (train) dimension
        dists = np.zeros((M, N)) # Initialize an MxN (500, 5000) array with values of 0
		
        test_sum_squares = np.sum(np.square(X), axis=1).reshape((M, 1)) # Sum of Squares of Test
        train_sum_squares = np.sum(np.square(self.X_train), axis=1).reshape((1, N)) # Sum of Squares of Train
        test_train = X.dot(self.X_train.T) # Dot or Inner Product in MxN size with transposition of arrays
        dists = np.sqrt(test_sum_squares - (2 * test_train) + train_sum_squares)
		
		# Formula Optimization Test
		# sqrt((x-y)^2) = sqrt((x-y)*(x-y)) = sqrt(x^2 - 2xy + y^2)
        # test_sum_squares = np.sum(X, axis=1).reshape((M, 1)) # Sum of Test
        # train_sum_squares = np.sum(self.X_train, axis=1).reshape((1, N)) # Sum of Train
        # dists = np.sqrt(np.square(test_sum_squares - train_sum_squares)) # sqrt((x-y)^2)
		
        return dists
