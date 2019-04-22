import numpy as np

X = np.array([0, 1, 2])
y = np.array([0, 1, 2])

class LinearModel():
    def __init__(self, coeff, intercept):
        self.coeff = coeff
        self.intercept = intercept
    
    def predict(self, X):
        pass

class LinearRegression(LinearModel):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass
    

