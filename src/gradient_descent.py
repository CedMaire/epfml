import numpy as np

"""
Subfunctions related to the gradient descent and stochastic gradient descent algorithms.
"""

def compute_gradient(y, tx, w):
    """
    Computes the gradient.

    :param y: expected labels vector
    :param tx: data matrix
    :param w: weight vector
    :returns: the gradient
    """

    y_len = len(y)

    error = y - np.dot(tx, w)
    gradient = -np.dot(tx.T, error) / y_len

    return gradient, error

def batch_iteration(y, tx):
    """
    Return a random batch of size 1.

    :param y: expected labels vector
    :param tx: matrix of samples and features
    :yields: random batch of size 1
    """

    rand = np.random.randint(len(y))
    return y[rand], tx[rand]
