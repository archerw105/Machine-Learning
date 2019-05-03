import numpy as np 

class Perceptron:
    def __init__(self, input_num):
        """
        Parameters
        ----------
        input_num: number of input variables for 1-layer neural network

        Notes
        -----
        0-th index reserved for bias value
        """
        self.weights = np.zeros(input_num+1)

    def train(self, training_data, l_rate, threshold):
        """Standard gradient descent

            Parameters
            ----------
            training_data: input matrix appended by target column vector
            l_rate: learning rate
            threshold: number of iterations (epochs)
        """    
        for _ in range(threshold):
            sum_error = 0.0
            for row in training_data:
                input = row[:-1]
                target = row[-1]
                error = self.predict(input) - target
                #cost = 1/2*(error)**2
                self.weights[0] -= error*l_rate
                for i in range(len(row)-1):
                    self.weights[i+1] -= error*row[i]*l_rate
        return self.weights

    def predict(self, input):
        """
        Parameters
        ----------
        input: input row vector
        """
        activation = np.dot(self.weights[1:], input) + self.weights[0]
        if activation >= 0:
            return 1
        else:
            return 0

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
l_rate = 0.1
n_epoch = 5
a = Perceptron(2)
weights = a.train(dataset, l_rate, n_epoch)
print(weights)