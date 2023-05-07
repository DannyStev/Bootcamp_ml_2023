import numpy as np
from plot_results import plot_optimizers, plot_optimizer
from sgd_optimizer import sgd
from adam_optimizer import adam
from rmsprop_optimizer import rmsprop


# Function to plot all the optimizer and compare them
X = np.random.randn(100, 2)
y = np.dot(X, [2 , 3]) + np.random.randn(100) * 0.1

# Calling sgd
w_sgd = sgd(X, y, alpha= 0.001, epochs=100, batch_size=32, print_every=5)

# Calling adam
w_adam = adam(X, y, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=100, batch_size=332, print_every=10)

# Calling rmsprop
w_rmsprop = rmsprop(X, y, alpha=0.001, beta=0.9, epsilon=1e-8, epochs=100, batch_size=32, print_every=10)

plot_optimizers(X, y, w_sgd, w_adam, w_rmsprop)

#Plot sgd
#plot_optimizer(X,y,w_sgd,"SGD")
#Plot adam
#plot_optimizer(X, y, w_adam, "ADAM")
#Plot rmsprop
#plot_optimizer(X, y, w_rmsprop, "RMS Prop")