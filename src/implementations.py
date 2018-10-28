import numpy as np
import math
from cost_computer import compute_loss
from gradient_descent import compute_gradient, batch_iteration

"""
All the requested implementations.
"""

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Computes the weights and the losses by applying the gradient descent algorithm.

    :param y: expected labels vector
    :param tx: data matrix
    :param initial_w: the initial weights to start with
    :param max_iters: the maximal number of iterations to apply
    :param gamma: the gamma factor to apply to the gradient
    :returns: w - computed weights
              loss - computed loss
    """

    w = initial_w
    loss = math.inf

    for i in range(max_iters):
        gradient, _ = compute_gradient(y, tx, w)

        w = w - gamma * gradient
        loss = compute_loss(y, tx, w)

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Computes the weights and the losses by applying the stochastic gradient descent algorithm with a batch size of 1.

    :param y: expected labels vector
    :param tx: data matrix
    :param initial_w: the initial weights to start with
    :param max_iters: the maximal number of iterations to apply
    :param gamma: the gamma factor to apply to the gradient
    :returns: w - computed weights
              loss - computed loss
    """

    w = initial_w
    loss = math.inf

    for i in range(max_iters):
        y_rand, tx_rand = batch_iteration(y, tx)

        gradient, _ = compute_gradient(y_rand, tx_rand, w)

        w = w - gamma * gradient
        loss = compute_loss(y, tx, w)

    return w, loss

def least_squares(y, tx):

    txTransposed = tx.T
    txDotT = np.dot(txTransposed, tx)
    txTdotY = np.dot(txTransposed, y)
    w = np.linalg.solve(txDotT, txTdotY)

    loss = np.sqrt(2 * compute_loss(y, tx, w))

    return w, loss

def ridge_regression(y, tx, lambda_):
    loss = None

    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = None
    loss = None

    raise NotImplementedError

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = None
    loss = None

    raise NotImplementedError

    return w, loss
