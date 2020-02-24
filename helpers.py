from numpy import ndarray, exp, zeros


def sigmoid(z: ndarray) -> ndarray:
    """The sigmoid function."""
    return 1.0 / (1.0 + exp(-z))


def sigmoid_prime(z: ndarray) -> ndarray:
    """Derivative of the sigmoid function."""
    y = sigmoid(z)
    return y * (1 - y)


def cost_derivative(output: ndarray, expected: ndarray) -> ndarray:
    """Return the vector of the cost function's partial derivatives in respect to the output"""
    return output - expected


def digitVector(digit: int) -> ndarray:
    """
    Return a (10,1) ndarray with all elements as ``0.0``,
    except the one in the index that equals the provided digit, which is set as ``1.0``
    """
    vector = zeros((10, 1))
    vector[digit] = 1.0
    return vector
