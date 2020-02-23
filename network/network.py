import numpy as np
from typing import List, Iterator, Tuple, Optional
from helpers import sigmoid, sigmoid_prime, cost_derivative, signal_last
from mnist.loader import Loader


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
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for y, x in zip(layers[1:], layers[:-1])]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """given ``a`` as input, Return the output of the network"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w.dot(a) + b)
        return a

    def SGD(
        self,
        training_data: Loader,
        epochs: int,
        mini_batch_size: int,
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
        mini_batch = []

        for j in range(epochs):

            for is_last_entry, entry in signal_last(training_data):

                mini_batch.append(entry)

                if len(mini_batch) == mini_batch_size or is_last_entry:

                    self.descend(mini_batch, learning_rate)
                    mini_batch.clear()

            if test_data:
                successful, total = self.evaluate(test_data)
                print(f"Epoch {j}: {successful} / {total}")
            else:
                print(f"Epoch {j} complete")

    def descend(self, batch: List[Tuple[np.ndarray, np.ndarray]], learning_rate: int):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation.
        ``batch`` is a list of tuples ``(in, out)`` representing the training inputs and the desired
        outputs.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for _in, expected_out in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(_in, expected_out)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - nw * (learning_rate / len(batch))
            for w, nw in zip(self.weights, nabla_w)
        ]

        self.biases = [
            b - nb * (learning_rate / len(batch)) for b, nb in zip(self.biases, nabla_b)
        ]

    def backpropagate(
        self, _in: np.ndarray, expected_out: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy ndarrays, similar
        to ``self.biases`` and ``self.weights``.
        """

        nabla_b = [None] * len(self.biases)
        nabla_w = [None] * len(self.weights)
        activation = _in
        activations = [_in]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = w.dot(activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # calculating the error in the output layer
        delta = cost_derivative(activations[-1], expected_out) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(activations[-2].T)

        # backpropagating the error to previous layers.
        for i in range(len(self.layers) - 2, 0, -1):
            delta = self.weights[i].T.dot(delta) * sigmoid_prime(zs[i - 1])
            nabla_b[i - 1] = delta
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
