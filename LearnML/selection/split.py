#split.py

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
     #first, the function converts the input data X and y into numpy arrays using np.asarray(). This ensures that the data is in a consistent format for further processing.
     X = np.asarray(X)
     y = np.asarray(y)
     #the function checks if the number of samples in X and y are the same. If they are not, it raises a ValueError indicating that X and y must have the same number of samples.
     if len(X) != len(y):
         raise ValueError("X and y must have the same number of samples.")
     #also the function checks if the test_size parameter is between 0 and 1. If it is not, it raises a ValueError indicating that test_size must be between 0 and 1.
     if not 0 < test_size < 1:
         raise ValueError("test_size must be between 0 and 1.")
     #second, the function calculates the number of samples in the dataset using len(X) and creates an array of indices from 0 to n_samples - 1 using np.arange(n_samples). This array of indices will be used to split the data into training and testing sets.
     n_samples = len(X)
     indices = np.arange(n_samples)
     #now, if the shuffle parameter is set to True, the function creates a random number generator using np.random.default_rng(random_state) and shuffles the indices array using rng.shuffle(indices). This randomizes the order of the samples before splitting them into training and testing sets.
     if shuffle:
         rng = np.random.default_rng(random_state)
         rng.shuffle(indices)
     #at last, the function calculates the number of samples to be included in the test set based on the test_size parameter. It then splits the indices array into test_indices and train_indices based on this count. Finally, it uses these indices to create the training and testing sets for both X and y, and returns them as X_train, X_test, y_train, and y_test.
     test_count = int(n_samples * test_size)
 
     test_indices = indices[:test_count]
     train_indices = indices[test_count:]
 
     X_train = X[train_indices]
     X_test = X[test_indices]
 
     y_train = y[train_indices]
     y_test = y[test_indices]
 
     return X_train, X_test, y_train, y_test