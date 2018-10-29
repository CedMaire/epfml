import numpy as np

"""
Subfunctions related to the logistic regression and regularized logistic regression algorithms.
"""

def logistic_function(z):
    """
    Logistic function, in our case this is the sigmoid function.

    :param z: a number
    :returns: the sigmoid function of the parameter z
    """

    return 1.0 / (1 + np.exp(-z))

def calculate_loss(y, tx, w):
    """
    Computes the loss for the [regularized] logistic regression using the logistic function.

    :param y: expected labels
    :param tx: data set
    :param w: weights vector
    :returns: generated loss
    """

    sigma = logistic_function(tx.dot(w))

    loss_1 = -np.matmul(y.T, (np.log(sigma)))
    loss_2 = np.matmul((1 - y).T, (np.log(1 - sigma)))

    return loss_1 + loss_2 / len(sigma)

def calculate_gradient(y, tx, w):
    """
    Computes the gradient used by the [regularized] logistic regression using the logistic function.

    :param y: expected labels
    :param tx: data set
    :param w: weights vector
    :returns: gradient
    """

    sigma = logistic_function(np.matmul(tx,w))

    z = sigma - y
    grad = np.matmul(tx.T, z)

    return grad / (len(sigma) / 2)
