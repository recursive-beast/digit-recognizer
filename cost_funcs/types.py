class CostFunction:
    """A class that represents a cost function"""

    def __call__(self, a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``"""
        raise NotImplementedError

    def delta(self, a, y, z):
        """Return the error delta from the output layer"""
        raise NotImplementedError
