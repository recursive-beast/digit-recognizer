import gzip
import pickle
import numpy as np
from helpers import resultVector
from typing import Tuple, Optional, Union


class Loader:

    """
    A reusable Iterator that yields tuples with two entries when its ``.__next__()`` is called :

    1. The first entry is a (784,1) ndarray that represents an MNIST data set image.
        784 is the number of pixels in each image (28 * 28 pixels).
        
    2. the second entry's structure depends on the vectorize_results parameter :

        if True
            it's a (10,1) ndarray representing the desired output for the corresponding input image.
        else
            it's the digit that corresponds to the input image.
    """

    def __init__(self, gzip_location: str, vectorize_results: Optional[bool] = False):
        self.file = gzip.open(gzip_location, "rb")
        self.vectorize_results = vectorize_results

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, Union[np.ndarray, int]]:
        try:
            entry = pickle.load(self.file)

            if self.vectorize_results:
                return (entry[0], resultVector(entry[1]))

            return entry

        except EOFError:
            self.file.seek(0)
            raise StopIteration()

    def __del__(self):
        self.file.close()

