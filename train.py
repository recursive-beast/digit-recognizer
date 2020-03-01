from mnist import Loader, MiniBatchLoader
from network import Network

learning_rate = 3.0
mini_batch_size = 10
epochs = 30

training = MiniBatchLoader(Loader("data/training.gz"), mini_batch_size)
testing = Loader("data/testing.gz")

net = Network([784, 30, 10])

net.SGD(training, epochs, learning_rate, testing)
