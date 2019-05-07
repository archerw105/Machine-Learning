import numpy as np
import random
import mnist_loader
#based off of Michael Nielsen's "Neural Networks and Deep Learning"
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

class Network: 
    def __init__(self, sizes):
        """
        Parameters
        ---------
        sizes: number of neurons in each layer
        num_layers: self-explanatory
        weights: 3-D matrix, weights[i][j][k] is the weight from j'th neuron 
                in (i+1)'th layer to k'th neuron in i'th layer
        biases: 

        Notes
        -----
        -First layer is input layer and has no bias values
        -ordering of j and k indices in weight matrix is swapped, but allows us to use matrix multiplication
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(l2, l1) for l1, l2 in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(l, 1) for l in sizes[1:]]
    
    def feedforward(self, input):
        """Returns output from input vector"""
        op = input
        for w, b in zip(self.weights, self.biases):
            op = sigmoid(np.dot(w, op) + b)
        return op

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """
        Parameters
        ----------
        training_data: tuples (x, y) representing training inputs and outputs
        mini_batch_size:
        eta:learning rate
        epochs:
        """
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for _ in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] 
                            for k in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.update_from_batch(batch, eta)
           
            if test_data:
                print("Epoch {} : {} / {}".format(_,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(_))

    def update_from_batch(self, batch, eta):
        """
        Updates weights and biases by applying gradient descent to the mini-batch

        Notes
        -----
        delta_nabla_w and delta_nabla_b are the gradients wrt w and b of the cost function
        nabla_w and nabla_b are the sum of the gradients wrt w and b
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]#nw and dnw are matrices
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        
        self.weights = [w - eta*(nw/len(batch))
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta*(nb/len(batch))
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Computes gradient wrt every weight and bias"""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #feedforward
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])#Hadamard product
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #creates matrix
        nabla_b[-1] = delta
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_b[-l] = delta

        return (nabla_w, nabla_b)
    
    def cost_derivative(self, output, y):
        """Cost function is set to be 1/2*(output - y)^2"""
        return (output - y)

    def evaluate(self, test_data):#copied
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

net = Network([784, 30, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net.sgd(training_data, 30, 10, 3.0, test_data=test_data)