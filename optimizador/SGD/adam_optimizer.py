# Install libraries
import numpy as np


def adam(X, y, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=100, batch_size=32, print_every=10):
    # Initialize the weights to small random values
    m, n = X.shape
    w = np.random.randn(n)

    # Initialize the first and second moment estimates
    m_t = np.zeros(n)
    v_t = np.zeros(n)

    # Run the Adam algorithm
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

            # Update the first and second moment estimates
            m_t = beta1 * m_t + (1 - beta1) * grad
            v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)

            # Compute the bias-corrected first and second moment estimates
            m_t_hat = m_t / (1 - beta1 ** (epoch * num_batches + i + 1))
            v_t_hat = v_t / (1 - beta2 ** (epoch * num_batches + i + 1))

            # Update the weights
            w -= alpha * m_t_hat / (np.sqrt(v_t_hat) + epsilon)

        # Print the loss every print_every epochs
        if epoch % print_every == 0:
            loss = np.sum((np.dot(X, w) - y) ** 2) / (2 * m)
            print(f"Epoch {epoch}, Loss {loss:.4f}")

    return w
