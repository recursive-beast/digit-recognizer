"""
cost_funcs
----------
A module that defines different cost functions for use in neural networks
"""


import numpy as np

from helpers import sigmoid_prime
from cost_funcs.types import CostFunction


class QuadraticCost(CostFunction):
    """A class that represents the quadratic cost function"""

    def __call__(self, a: np.ndarray, y: np.ndarray) -> float:
        """Return the cost associated with an output ``a`` and desired output ``y``"""
        return 0.5 * np.linalg.norm(a - y) ** 2

    def delta(self, a: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the error delta from the output layer"""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(CostFunction):
    """A class that represents the quadratic cost function"""

    def __call__(self, a: np.ndarray, y: np.ndarray) -> float:
        """Return the cost associated with an output ``a`` and desired output ``y``"""
        # Note that np.nan_to_num is used to ensure numerical stability.
        # In particular, if both ``a`` and ``y`` have a 1.0 in the same slot,
        # then the expression (1-y)*np.log(1-a) returns nan.
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def delta(self, a: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the error delta from the output layer"""
        return a - y
