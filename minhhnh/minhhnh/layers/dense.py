import numpy as np
from .layer import Layer
class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.01):
        self.learning_rate = learning_rate
        # initialize weights with small random numbers. We use normal initialization
        np.random.seed(0)
        self.weights = np.random.randn(input_units, output_units)*0.1
        # scale = 1/max(1., (2+2)/2.)
        # limit = math.sqrt(3.0 * scale)
        # self.weights = np.random.uniform(-limit, limit, size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, input):
        return np.matmul(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, np.transpose(self.weights))

        # compute gradient w.r.t. weights and biases
        grad_weights = np.transpose(np.dot(np.transpose(grad_output), input))
        grad_biases = np.sum(grad_output, axis=0)

        # Here we perform a stochastic gradient descent step.
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input
