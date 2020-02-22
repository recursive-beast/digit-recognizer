import gzip
import pickle
import numpy as np
from typing import Iterator, Tuple, Optional, Union


def resultVector(digit: int) -> np.ndarray:
    """
    Return a (10,1) ndarray with all elements as 0.0,
    except the one in the index that equals the provided digit
    """
    vector = np.zeros((10, 1))
    vector[digit] = 1.0
    return vector


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

        entry = pickle.load(file)

        if vectorize_results:
            yield (entry[0], resultVector(entry[1]))
        else:
            yield entry

