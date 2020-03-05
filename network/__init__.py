"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.

Gradients are calculated using backpropagation.
"""


import numpy as np

from helpers import sigmoid, sigmoid_prime, unitVector
from mnist import LazyDataLoader, TrainingDataLoader, mini_batches

from cost_funcs.types import CostFunction
from typing import Iterator, List, Optional, Tuple, Union

Loader = Iterator[Tuple[np.ndarray, Union[np.ndarray, int]]]


class Network(object):
    def __init__(self, layers: List[int], cost: CostFunction) -> None:
        """
        The list ``layers`` contains the number of neurons in the
        respective layers of the network.

        - example :
        if the list was [2, 3, 1] then it would be a three-layer network with the first layer containing 2 neurons,
        second layer 3 neurons, and the third layer 1 neuron.

        The biases and weights for the network are initialized randomly.

        Note that the first layer is assumed to be an input layer, and by convention we won't set any biases for those neurons
        """

        self.layers = layers

        self.cost = cost

        rng = np.random.default_rng()

        self.biases = [rng.standard_normal((y, 1)) for y in layers[1:]]

        self.weights = [
            rng.standard_normal((y, x)) / np.sqrt(x)
            for y, x in zip(layers[1:], layers[:-1])
        ]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """given ``a`` as input, Return the output of the network"""

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w.dot(a) + b)

        return a

    def SGD(
        self,
        training_data_location: str,
        epochs: int,
        mini_batch_size: int,
        learning_rate: float,
        lmbda: float,
        evaluation_data_location: str = None,
        monitor_evaluation_accuracy: bool = False,
        monitor_evaluation_cost: bool = False,
        monitor_training_accuracy: bool = False,
        monitor_training_cost: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Train the neural network using stochastic gradient descent and L2 regularization .

        the cost and accuracy on either the evaluation data or the training data can be monitored by setting the appropriate flags .

        The method returns a tuple containing four lists containing the accuracy and cost monitored for each epoch .

        This is useful for tracking progress, but slows things down substantially.
        """

        training_data = TrainingDataLoader(training_data_location)
        evaluation_data = LazyDataLoader(evaluation_data_location)

        training_data_size = len(training_data)

        training_accuracy, training_cost = [], []
        evaluation_accuracy, evaluation_cost = [], []

        for j in range(epochs):

            for X, Y in mini_batches(training_data, mini_batch_size):
                self.descend(
                    X, Y, mini_batch_size, training_data_size, learning_rate, lmbda
                )

            print(f"\nEpoch {j} complete -----------\n")

            if monitor_evaluation_accuracy:
                successful, total = self.accuracy(evaluation_data)
                evaluation_accuracy.append(successful / total * 100)
                print(f"Accuracy on evaluation data: {successful} / {total}")

            if monitor_evaluation_cost:
                total_cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(total_cost)
                print(f"Cost on evaluation data: {total_cost}")

            if monitor_training_accuracy:
                successful, total = self.accuracy(training_data)
                training_accuracy.append(successful / total * 100)
                print(f"Accuracy on training data: {successful} / {total}")

            if monitor_training_cost:
                total_cost = self.total_cost(training_data, lmbda)
                training_cost.append(total_cost)
                print(f"Cost on training data: {total_cost}")

        return (
            evaluation_accuracy,
            evaluation_cost,
            training_accuracy,
            training_cost,
        )

    def descend(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        mini_batch_size: int,
        training_data_size: int,
        learning_rate: float,
        lmbda: float,
    ) -> None:
        """
        Update the network's weights and biases by applying gradient descent using backpropagation.

        - ``X`` is a column stack where each column is representing a 28 * 28 image
        - ``Y`` is a column stack where each column is representing the desired output for the corresponding ``X`` column
        """

        nabla_b, nabla_w = self.backpropagate(X, Y)

        scale = learning_rate / mini_batch_size
        weight_decay = 1 - learning_rate * (lmbda / training_data_size)

        self.weights = [
            weight_decay * w - scale * nw for w, nw in zip(self.weights, nabla_w)
        ]

        self.biases = [b - scale * nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the sum of the
        gradient for the cost function over all training inputs in the provided mini_batch.
        
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy ndarrays, similar
        to ``self.biases`` and ``self.weights``.
        """

        nabla_b = [None] * len(self.biases)
        nabla_w = [None] * len(self.weights)
        activation = X
        activations = [X]  # list to store all the activations, layer by layer
        Zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            Z = w.dot(activation) + b
            Zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)

        # calculating the error in the output layer
        delta = self.cost.delta(activations[-1], Y, Zs[-1])

        # summing over all training inputs in the mini batch
        nabla_b[-1] = delta.sum(1, keepdims=True)

        # because of matrix multiplication, there's no need to sum over training inputs
        nabla_w[-1] = delta.dot(activations[-2].T)

        # backpropagating the error to previous layers.
        for i in range(len(self.layers) - 2, 0, -1):
            delta = self.weights[i].T.dot(delta) * sigmoid_prime(Zs[i - 1])
            nabla_b[i - 1] = delta.sum(1, keepdims=True)
            nabla_w[i - 1] = delta.dot(activations[i - 1].T)

        return (nabla_b, nabla_w)

    def accuracy(self, data: Loader) -> Tuple[int, int]:
        """
        Return an (int,int) tuple where:
        
        - the first entry is the number of test inputs for which the neural network outputs the correct result.
        - the second entry is the total number of test inputs
        """

        successful = 0
        total = 0
        for x, y in data:
            total += 1

            if type(y) == np.ndarray:
                y = y.argmax()

            if self.feedforward(x).argmax() == y:
                successful += 1

        return (successful, total)

    def total_cost(self, data: Loader, lmbda: float):
        """Return the total cost for the data set ``data``"""

        total_cost = 0.0
        data_size = 0

        for x, y in data:
            data_size += 1
            a = self.feedforward(x)

            if type(y) == int:
                y = unitVector(y)

            total_cost += self.cost(a, y)

        total_cost += 0.5 * lmbda * sum(np.linalg.norm(w) ** 2 for w in self.weights)

        return total_cost / data_size
