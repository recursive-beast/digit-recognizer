import gzip
import pickle
import numpy as np
from random import Random
from time import time
from helpers import digitVector
from typing import Tuple, List


class Loader:

    """
    A reusable Iterator that yields tuples with two entries when its ``.__next__()`` is called :

    1. The first entry is a (784,1) ndarray that represents an MNIST data set image.
        784 is the number of pixels in each image (28 * 28 pixels).
        
    2. the second entry is the digit that corresponds to the input image.
    """

    def __init__(self, gzip_location: str):
        self.file = gzip.open(gzip_location, "rb")

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, int]:
        try:
            return pickle.load(self.file)
        except EOFError:
            self.file.seek(0)
            raise StopIteration

    def __del__(self):
        self.file.close()


class MiniBatchLoader:

    """
    A reusable Iterator that yields mini batches as tuples ``(X,Y)``.

    - ``X`` is a column stack where each column is representing a 28 * 28 image
    - ``Y`` is a column stack where each column is representing the desired output for the corresponding ``X`` column
    """

    def __init__(self, data: Loader, mini_batch_size: int):
        self.X = []  # inputs
        self.Y = []  # expected outputs
        for x, y in data:
            self.X.append(x)
            self.Y.append(digitVector(y))

        t = time()

        self.rngX = Random(t)
        self.rngY = Random(t)

        self.shuffleData()

        self.mini_batch_size = mini_batch_size
        self.cursor = 0

    def shuffleData(self):
        self.rngX.shuffle(self.X)
        self.rngY.shuffle(self.Y)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        limit = self.cursor + self.mini_batch_size

        if limit > len(self.X):
            self.shuffleData()
            self.cursor = 0
            raise StopIteration

        mini_batch_X = np.column_stack((self.X[self.cursor : limit]))
        mini_batch_Y = np.column_stack((self.Y[self.cursor : limit]))

        self.cursor = limit

        return (mini_batch_X, mini_batch_Y)
