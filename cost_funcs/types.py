import numpy as np


class CostFunction:
    """A class that represents a cost function"""

    def __call__(self, a: np.ndarray, y: np.ndarray) -> float:
        """Return the cost associated with an output ``a`` and desired output ``y``"""
        raise NotImplementedError

    def delta(self, a: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the error delta from the output layer"""
        raise NotImplementedError
