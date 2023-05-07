import numpy as np


# Function of SGD Optimizer
def sgd(X, y, alpha=0.01, epochs=100, batch_size=32, print_every=10):
    # Initialize the weights to small random values
    m, n = X.shape
    w = np.random.randn(n)

    # Run the SGD algorithm
    for epoch in range(epochs):
        # Shuffle the data
        perm = np.random.permutation(m)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        # Split the data into batches
        num_batches = m // batch_size
        for i in range(num_batches):
            # Get the current batch
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Compute the gradient of the loss function
            grad = np.dot(X_batch.T, np.dot(X_batch, w) - y_batch) / batch_size

            # Update the weights
            w -= alpha * grad

        # Print the loss every print_every epochs
        if epoch % print_every == 0:
            loss = np.sum((np.dot(X, w) - y) ** 2) / (2 * m)
            print(f"Epoch {epoch}, Loss {loss:.4f}")

    return w
