import gzip
import pickle
import numpy as np
from helpers import resultVector
from typing import Iterator, Tuple, Optional, Union


def load(
    data_type: str, vectorize_results: Optional[bool] = False
) -> Iterator[Tuple[np.ndarray, Union[np.ndarray, int]]]:
    """
    Return generator object that yields tuples with two entries :

    1. The first entry is a (784,1) ndarray that represents an MNIST data set image.
        784 is the number of pixels in each image (28 * 28 pixels).
        
    2. the second entry's structure depends on the vectorize_results parameter :

        if True
            it's a (10,1) ndarray representing the desired output for the corresponding input image.
        else
            it's the digit that corresponds to the input image.
    """

    with gzip.open("data/" + data_type + ".gz") as file:

        try:
            while True:

                entry = pickle.load(file)

                if vectorize_results:
                    yield (entry[0], resultVector(entry[1]))
                else:
                    yield entry

        except EOFError:
            raise StopIteration

