from mnist import Loader, MiniBatchLoader
from network import Network
from cost_funcs import QuadraticCost

mini_batch_size = 10
epochs = 30
learning_rate = 0.5
lmbda = 5.0

training = MiniBatchLoader(Loader("mnist/data/training.gz"), mini_batch_size)
testing = Loader("mnist/data/testing.gz")
cost = QuadraticCost()

net = Network([784, 30, 10], cost)

net.SGD(training, epochs, learning_rate, lmbda, testing)
