import numpy as np
from numpy.core.tests.test_mem_overlap import xrange


class NearestNeighbor:
    def __init__(self):
        pass
# data set X is given, y denotes the labels.
# here the class is given data to remember but nothing is being done
# "assign to the class instance methods"

    def train(self, X, y):
        # X is N * D where each row is an example. Y is 1-dimension of size N'''
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y
# At predict time, new test set of images X with a for loop that goes over every test image independantly
# This gets the distance to every single  training image
    # this is only one line of vectorized python code ( distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1))
    # In this single line of code we are comparing a test image to every single training image in the database
    # and we are computing the distance as formulated by the L1/Manhattan distance
    # Vecotrised code takes away the need for multiple complex four loops
# Next instances that are closest are compared (min_index)
    # min_index is the index of the training example that is the lowest distance 
# Then we are predicting for this image the label of whatever was nearest. ( Ypred[i] = self.ytr[min_index] )

    def predict(self, X):
        # X is N * D where each row is an example we wish to predict label for
        num_test = X.shape[0]
        # lets make sure the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all the test rows
        for i in xrange(num_test):
            # find the nearest training image to the i^th test image
            # using the L1 distance/Manhattan (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

        return Ypred
