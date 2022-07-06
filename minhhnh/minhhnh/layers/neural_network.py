from .dense import Dense, Layer
from .relu import ReLU

class NeuralNetwork(Layer):
    def __init__(self, layers):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        self.network = []
        for i in range(len(layers)-1):
            self.network.append(Dense(layers[i], layers[i+1]))
            self.network.append(ReLU())

    def forward(self, X):
        """
        Compute activations of all network layers by applying them sequentially.
        Return a list of activations for each layer. 
        Make sure last activation corresponds to network logits.
        """
        activations = []
        input = X
        for i in range(len(self.network)):
            activations.append(self.network[i].forward(X))
            X = self.network[i].forward(X)

        assert len(activations) == len(self.network)
        return activations

    def predict(self, X):
        """
        Compute network predictions.
        """
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)

    def backward(self, input, grad_output):
        pass

    def __len__(self):
        return len(self.network)

    def __getitem__(self, index):
        return self.network[index]
