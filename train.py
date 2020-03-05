from network import Network
from cost_funcs import CrossEntropyCost
from matplotlib import gridspec, pyplot

epochs = 60
mini_batch_size = 10
learning_rate = 0.1
lmbda = 5.0

cost = CrossEntropyCost()

net = Network([784, 100, 10], cost)

ea, ec, ta, tc = net.SGD(
    "mnist/data/training.gz",
    epochs,
    mini_batch_size,
    learning_rate,
    lmbda,
    "mnist/data/testing.gz",
    True,
    True,
    True,
    True,
)

x = list(range(epochs))

gs = gridspec.GridSpec(3, 2)

pyplot.figure()

# ----------------------------
acc_plot = pyplot.subplot(gs[0, :]).set_xticklabels([])

pyplot.plot(x, ea, label="evaluation accuracy")
pyplot.plot(x, ta, label="training accuracy")

pyplot.ylabel("accuracy")

pyplot.legend()

# ----------------------------

pyplot.subplot(gs[1, :]).set_xticklabels([])

pyplot.plot(x, ec, label="evaluation cost")

pyplot.ylabel("cost")

pyplot.legend()
# ----------------------------

pyplot.subplot(gs[2, :])

pyplot.plot(x, tc, label="training cost")

pyplot.xlabel("epochs")
pyplot.ylabel("cost")

pyplot.legend()
# ----------------------------

pyplot.show()
pyplot.close()
