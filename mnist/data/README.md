# MNIST data set

each gzip file in this directory contains tuples (serialized with python's pickle module) of two entries :

- the first entry is an ndarray of floats with a shape of (784,1) representing the grayscale value of each pixel in a 28 x 28 image (1.0 for black, and 0.0 for white)
- the second entry is an int representing the desired output from the neural network if the first entry is used as the input.