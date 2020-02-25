import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    y = sigmoid(z)
    return y * (1 - y)


def cost_derivative(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the vector of the cost function's partial derivatives in respect to the output ``a``"""
    return a - y


def unitVector(i: int) -> np.ndarray:
    """
    Return a (10,1) ndarray with all elements as ``0.0``, except in the ``i``th index where the value is ``1.0``.
    """
    vector = np.zeros((10, 1))
    vector[i] = 1.0
    return vector
