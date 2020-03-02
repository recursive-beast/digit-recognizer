from mnist import Loader, MiniBatchLoader
from network import Network
from cost_funcs import QuadraticCost

learning_rate = 3.0
mini_batch_size = 10
epochs = 30

training = MiniBatchLoader(Loader("mnist/data/training.gz"), mini_batch_size)
testing = Loader("mnist/data/testing.gz")
cost = QuadraticCost()

net = Network([784, 30, 10], cost)

net.SGD(training, epochs, learning_rate, testing)
