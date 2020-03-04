from network import Network
from cost_funcs import QuadraticCost
import matplotlib.pyplot as plt

epochs = 30
mini_batch_size = 10
learning_rate = 0.25
lmbda = 5.0

cost = QuadraticCost()

net = Network([784, 30, 10], cost)

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

fig, (ax1, ax2) = plt.subplots(2, 1, True)

# ----------------------------
ax1.plot(x, ea, label="evaluation accuracy")
ax1.plot(x, ta, label="training accuracy")

ax1.set_xlabel("epochs")
ax1.set_ylabel("accuracy")

ax1.legend()
# ----------------------------
ax2.plot(x, ec, label="evaluation cost")
ax2.plot(x, tc, label="training cost")

ax2.set_xlabel("epochs")
ax2.set_ylabel("total cost")

ax2.legend()
# ----------------------------

plt.show()
plt.close()
