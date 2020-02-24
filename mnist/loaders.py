import gzip
import pickle
import numpy as np
from random import shuffle
from helpers import resultVector
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
    A reusable Iterator that yields mini batches (lists) of tuples.
    the tuples are extracted from the provided loader.
    """

    def __init__(self, data: Loader, mini_batch_size: int):
        self.data = []
        for _input, digit in data:
            self.data.append((_input, resultVector(digit)))

        shuffle(self.data)

        self.mini_batch_size = mini_batch_size
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        limit = self.cursor + self.mini_batch_size

        if limit > len(self.data):
            shuffle(self.data)
            self.cursor = 0
            raise StopIteration

        mini_batch = self.data[self.cursor : limit]

        self.cursor = limit

        return mini_batch
