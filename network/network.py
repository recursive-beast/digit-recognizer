import numpy as np
from typing import List, Iterator, Tuple, Optional
from helpers import sigmoid, sigmoid_prime, cost_derivative
from mnist.loaders import Loader, MiniBatchLoader


class Network(object):
    def __init__(self, layers: List[int]):
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
        rng = np.random.default_rng()
        self.biases = [rng.standard_normal((y, 1)) for y in layers[1:]]
        self.weights = [
            rng.standard_normal((y, x)) for y, x in zip(layers[1:], layers[:-1])
        ]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """given ``a`` as input, Return the output of the network"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w.dot(a) + b)
        return a

    def SGD(
        self,
        training_data: MiniBatchLoader,
        epochs: int,
        learning_rate: float,
        test_data: Optional[Loader] = None,
    ):
        """
        Train the neural network using stochastic gradient descent.
        
        - ``training_data`` is an iterator over tuples ``(in, out)`` representing the training inputs and the desired
        outputs.
        
        - If ``test_data`` is provided then the network will be evaluated against the test data after each epoch, and partial progress will be printed out.
        This is useful for tracking progress, but slows things down substantially.
        """

        for j in range(epochs):

            for mini_batch_X, mini_batch_Y in training_data:
                self.descend(
                    mini_batch_X,
                    mini_batch_Y,
                    training_data.mini_batch_size,
                    learning_rate,
                )

            if test_data:
                successful, total = self.evaluate(test_data)
                print(f"Epoch {j}: {successful} / {total}")
            else:
                print(f"Epoch {j} complete")

    def descend(
        self, X: np.ndarray, Y: np.ndarray, mini_batch_size: int, learning_rate: float
    ):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation.

        - ``X`` is a column stack where each column is representing a 28 * 28 image
        - ``Y`` is a column stack where each column is representing the desired output for the corresponding ``X`` column
        """

        nabla_b, nabla_w = self.backpropagate(X, Y)

        scale = learning_rate / mini_batch_size

        self.weights = [w - scale * nw for w, nw in zip(self.weights, nabla_w)]

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
        delta = cost_derivative(activations[-1], Y) * sigmoid_prime(Zs[-1])

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

    def evaluate(self, test_data: Loader) -> Tuple[int, int]:
        """
        Return an (int,int) tuple where:
        
        - the first entry is the number of test inputs for which the neural network outputs the correct result.
        - the second entry is the total number of test inputs
        """

        counter = 0
        total = 0
        for _in, expected_out in test_data:
            total += 1
            if self.feedforward(_in).argmax() == expected_out:
                counter += 1

        return (counter, total)
