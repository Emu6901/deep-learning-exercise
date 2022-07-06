import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from layers.neural_network import NeuralNetwork
class Trainer:
    def __init__(self):
        pass

    def softmax_crossentropy_with_logits(self, logits, reference_answers):
        """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
        logits_for_answers = logits[np.arange(len(logits)), reference_answers]

        xentropy = -logits_for_answers * \
            np.log(np.sum(np.exp(logits), axis=-1))

        return xentropy

    def grad_softmax_crossentropy_with_logits(self, logits, reference_answers):
        """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)), reference_answers] = 1

        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

        return (-ones_for_answers + softmax) / logits.shape[0]

    def train(self, network, X, y):
        """
        Train your network on a given batch of X and y.
        You first need to run forward to get all layer activations.
        Then you can run layer.backward going from last to first layer.
        After you called backward for all layers, all Dense layers have already made one gradient step.
        """

        # Get the layer activations
        layer_activations = network.forward(X)
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = self.softmax_crossentropy_with_logits(logits, y)
        loss_grad = self.grad_softmax_crossentropy_with_logits(logits, y)

        for i in range(1, len(network)):
            loss_grad = network[len(
                network) - i].backward(layer_activations[len(network) - i - 1], loss_grad)

        return np.mean(loss)

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            np.random.seed(0)
            indices = np.random.permutation(len(inputs))
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def fit(self, x_train, y_train, x_test, y_test, batchsize=16, n_splits=8):
        best_test_acc = 0
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        layers = [x_train.shape[1], 50, 100, 200,
                  400, 800, 1600, 800, 400, 200, 100, 50, 2]
        for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(x_train, y_train)):
            network = NeuralNetwork(layers)
            train_log = []
            val_log = []
            tqdm_object = tqdm(range(100), total=100)
            for _ in tqdm_object:

                for x_batch, y_batch in self.iterate_minibatches(
                    x_train[train_ids], y_train[train_ids], batchsize=batchsize, shuffle=True
                ):
                    self.train(network, x_batch, y_batch)

                train_log.append(np.mean(network.predict(
                    x_train[train_ids]) == y_train[train_ids]))
                val_log.append(np.mean(network.predict(
                    x_train[val_ids]) == y_train[val_ids]))

                test_acc = np.mean(network.predict(x_test) == y_test)
                if test_acc >= best_test_acc:
                    best_test_acc = test_acc
                    pickle.dump(
                        network,
                        open("best_network.pkl", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                tqdm_object.set_postfix(
                    train_acc=train_log[-1],
                    val_acc=val_log[-1],
                    best_test_acc=best_test_acc,
                    test_acc=test_acc,
                )
            print("Fold ", fold_idx)
            plt.plot(train_log, label="train accuracy")
            plt.plot(val_log, label="val accuracy")
            plt.legend(loc="best")
            plt.grid()
            plt.show()
