
import numpy as np
import matplotlib.pyplot as plt

# Plot all results
def plot_optimizers(X,y, w_sgd, w_adam, w_rmsprop):

    plt.scatter(X[:,0],y, label="Ground Truth")
    plt.scatter(X[:,0], np.dot(X, w_sgd), label="SGD")
    plt.scatter(X[:, 0], np.dot(X, w_adam), label="Adam")
    plt.scatter(X[:, 0], np.dot(X, w_rmsprop), label="RMSProp")
    plt.legend()
    plt.show()

def plot_optimizer(X,y, w, name="Default"):

    plt.scatter(X[:,0],y, label="Ground Truth")
    plt.scatter(X[:,0], np.dot(X, w), label=name)

    plt.legend()
    plt.show()
