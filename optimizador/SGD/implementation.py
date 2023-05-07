# Install libraries
import matplotlib.pyplot as plt
import numpy as np


# Implementation of SGD
def sgd(X,y,alpha=0.01, epochs=50,batch_size=32,print_every=10):

    # Initialize  the weights to small random values
    m,n = X.shape
    w = np.random.rand(n)
    for epoch in range(epochs):
        #shuffle data
        perm = np.random.permutation(m)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        #split the data into batches
        num_batches = m//batch_size
        for i in range(num_batches):
            # get the current batches
            start = i*batch_size
            end = start+batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # compute the gradient of the lost function
            grad = np.dot(X_batch.T, np.dot(X_batch.w)-y_batch)/batch_size


            # update the weights

            w -= alpha*grad

            #print the loss every print_every epoch
            if epoch % print_every == 0:
                loss = np.sum((np.dot(X,w)-y)**2)/(2*m)
                print(f"Epoch{epoch}, Loss{loss:.4f}")



    return w



# Implementation of Adam optimizer

def adam(X, y, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, epoch=50, batch_size=332, print_every=10):

    #initialize the weights to small random values
    m, n = X.shape
    w = np.random.randn(n)

    #initialize the first and second moment estimates
    m_t = np.zeros(n)
    v_t = np.zeros(n)

    #run the Ada, algorithm for epoch in range(epochs)
    #Shuffle the data
    perm = np.random.permutation(m)
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    #Split the data into batches
    num_batches = m // batch_size
    for i in range(num_batches):

        #Get the current batch
        start = i*batch_size
        end = start+batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        #compute the gradient of the loss function
        grad = np.dot(X_batch.T, np.dot(X_batch, w)-y_batch)/batch_size

        #update the first and second moment estimates
        m_t = beta1*m_t+(1-beta1)*grad
        v_t = beta2*v_t+(1-beta2)*(grad 2)

        #compute the bias-corrected forst and second moment estimates
        m_t_hat = m_t/(1-beta1(epoch*num_batches+i+1))
        v_t_hat = v_t/(1-beta2(epoch*num_batches+i+1))

        #Update the weights
        w-=alpha*m_t_hat/(np.sqrt(v_t_hat)+epsilon)

        #Print the loss every print_every epochs
        if epoch % print_every==0:
            loss = np.sum((np.dot(X,w)-y)2)/(2*m)
            print(f"Epoch {epoch}, Loss:.4f}")


        return w

# Implementation of Rmsprop
def rmsprop(X, y, alpha=0.001, beta=0.9, epsilon=1e-8, epoch=100, batch_size=32, print_every=10):

    # Initialize the weights to small random values
    m,n=X.shape
    w=np.random.randn(n)

    # Initialize the squared gradient estimate
    g_t=np.zeros(n)

    # Run the RMSProp algorithm
    for epoch in range(epochs):
        # Shuffle the data
    perm = np.random.permutation(m)
    X_shuffled = X[perm]
    Y_shuffled = y[perm]

    # Split the data into batches
    num_batches = m // batch_size
    for i in range(num_batches):
        # Get the current batch
        start = i * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Compute the gradient of the loss function
        grad = np.dot(X_batch.T, np.dot(X_batch, w) - y_batch)/ batch_size

        # Update the squared gradient estimate
        g_t = beta * grad /(np.sqrt(g_t) + epsilon)

        # Print the loss every print_every epochs
        if epoch % print_every == 0:
            loss = np.sum((np.dot(X, w) - y) 2) / (2 * m)
            print(f"Epoch {epoch}, Loss {loss:.4f}")

    return w



# Function to plot all the optimizer and compare them
X = np.random.randn(100,2)
y = np.dot(X,[2,3]) + np.random.randn(100) * 0.1

# Train model using SGD

w_sgd = sgd(X, y, alpha= 0.001, beta=0.9, epsilon=1e-8, epochs=25, batch_size=32, print_every=5)
# Train model using Adam
w_adam = adam(X, y, alpha= 0.001, beta=0.9, epsilon=1e-8, epochs=25, batch_size=32, print_every=5)
# Train model using rmsprop
w_rmsprop = rmsprop(X, y, alpha= 0.001, beta=0.9, epsilon=1e-8, epochs=25, batch_size=32, print_every=5)


# Plot results
def plot_optimizers(X,y, w_sgd, w_adam, w_rmsprop):

    plt.scatter(X[:,0],y, label="Ground Truth")
    plt.scatter(X[:,0], np.dot(X, w_sgd), label="SGD")
    plt.scatter(X[:, 0], np.dot(X, w_adam), label="Adam")
    plt.scatter(X[:, 0], np.dot(X, w_rmsprop), label="RMSProp")
    plt.legend()
    plt.show()

plot_optimizers(X,y,w_sgd, w_adam, w_rmsprop)




