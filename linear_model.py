import numpy as np

X = np.array([[0, 1], [1, 0], [2, -1]])
y = np.array([1, 2, 3])

class LinearModel():
    """Linear model from 
        coefficient and intercept parameters"""
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
        "Ax = y; A is training set appended by column of 1's; y is target vector"
        A = np.vstack([X.T, np.ones(len(X))]).T
        m = np.linalg.pinv((A.T).dot(A)).dot(A.T).dot(y)
        
        self.set_coeff(m[0:len(m)-1])
        self.set_intercept(m[len(m)-1])
        
        
lm = LinearRegression()

lm.fit(X, y)
print(lm.predict(X))