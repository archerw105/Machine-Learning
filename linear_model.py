import numpy as np
from sklearn import linear_model

X = np.array([[0, 1], [1, 1], [2, 1]])
y = np.array([0, 1, 2])

class LinearModel():
    """Linear model from 
        coefficient parameters and
        intercept parameter"""

    def __init__(self):
        self.coeff = np.array([])
        self.intercept = 0

    def predict(self, X):
        output = np.zeros(shape=(len(X)))
        for i in range(len(X)):
            output[i] = np.dot(self.coeff, X[i]) + self.intercept
        return output

    def set_coeff(self, coeff):
        self.coeff = coeff
    
    def set_intercept(self, intercept):
        self.intercept = intercept


class LinearRegression(LinearModel):
    """Linear regression from training set and target values"""
    def __init__(self):
        LinearModel.__init__(self)

    def fit(self, X, y):
        
        pass
        
lm = LinearRegression()

lm.set_coeff(np.array([1, 1]))
op = lm.predict(X)