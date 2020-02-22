from numpy import ndarray, exp
from typing import Iterator, Any, Tuple


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


def signal_last(it: Iterator[Any]) -> Iterator[Tuple[bool, Any]]:
    """
    Return an iterator that yields tuples of two values :
    - the first one is a boolan, ``True`` if the currently yielded tuple is the last one, ``False`` otherwise.
    - the second value is whatever the iterator ``it`` yields when ``next(it)`` is called.
    """

    ret_val = next(it)

    for val in it:
        yield False, ret_val
        ret_val = val

    yield True, ret_val
