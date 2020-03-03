from network import Network
from cost_funcs import QuadraticCost

epochs = 30
mini_batch_size = 10
learning_rate = 0.25
lmbda = 5.0

cost = QuadraticCost()

net = Network([784, 30, 10], cost)

evaluation_accuracy, evaluation_cost, training_accuracy, training_cost = net.SGD(
    "mnist/data/training.gz",
    epochs,
    mini_batch_size,
    learning_rate,
    lmbda,
    "mnist/data/testing.gz",
    True,
)
