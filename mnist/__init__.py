"""
mnist
~~~~~

A module to load the MNIST data set.
"""


import gzip
import numpy as np
import pickle

from random import shuffle
from typing import Tuple

from helpers import unitVector


class LazyDataLoader:

    """
    A reusable Iterator that lasily yields tuples with two entries when its ``.__next__()`` is called :

    1. The first entry is a (784,1) ndarray that represents an MNIST data set image.
        784 is the number of pixels in each image (28 * 28 pixels).
        
    2. the second entry is the digit that corresponds to the input image.

    the data is loaded lazily into memory.
    """

    def __init__(self, gzip_location: str) -> None:
        self.file = gzip.open(gzip_location, "rb")

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, int]:
        try:
            return pickle.load(self.file)
        except EOFError:
            self.file.seek(0)
            raise StopIteration

    def __del__(self) -> None:
        self.file.close()


class TrainingDataLoader:
    """
    A reusable Iterator that yields tuples with two entries when its ``.__next__()`` is called :

    1. The first entry is a (784,1) ndarray that represents an MNIST data set image.
        784 is the number of pixels in each image (28 * 28 pixels).
        
    2. the second entry is a numpy unit vector that represents the desired output from the network.

    the data is loaded eagerly into memory.
    """

    def __init__(self, gzip_location: str) -> None:
        self.data = []

        for x, y in LazyDataLoader(gzip_location):

            self.data.append((x, unitVector(y)))

        shuffle(self.data)

        self.cursor = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:

        if self.cursor >= len(self.data):
            shuffle(self.data)
            self.cursor = 0
            raise StopIteration

        val = self.data[self.cursor]

        self.cursor += 1

        return val


def mini_batches(training_data: TrainingDataLoader, mini_batche_size: int):
    """
    get an iterator over mini batches from the provided training data loader
    
    the iterator yields tuples with two entries when its ``.__next__()`` is called :

    - the first entry is a numpy column stack where each column is representing a 28 * 28 image
    - the second entry is a numpy column stack where each column is representing the desired output for the corresponding first entry column
    """

    X = []
    Y = []

    for x, y in training_data:

        X.append(x)
        Y.append(y)

        if len(X) == mini_batche_size:
            yield (np.column_stack((X)), np.column_stack((Y)))
            X = []
            Y = []

