import numpy as np
import math
from cost_computer import compute_loss
from gradient_descent import compute_gradient, batch_iteration
from logistic_regression import calculate_gradient, calculate_loss

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
    """
    Computes the weights and the losses by applying the least_squares algorithm.

    :param y: expected labels vector
    :param tx: data matrix
    :returns: w - computed weights
              loss - computed loss
    """

    txTransposed = tx.T
    txDotT = np.dot(txTransposed, tx)
    txTdotY = np.dot(txTransposed, y)

    w = np.linalg.solve(txDotT, txTdotY)
    loss = np.sqrt(2 * compute_loss(y, tx, w))

    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Computes the weights and the losses by applying the ridge regression algorithm.

    :param y: expected labels vector
    :param tx: data matrix
    :param lambda_: the lambda factor to apply to the identity matrix
    :returns: w - computed weights
              loss - computed loss
    """

    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = np.sqrt(2 * compute_loss(y, tx, w))

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Computes the weights and the losses by applying the logistic regression algorithm.

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
        grad = calculate_gradient(y, tx, w)

        loss = calculate_loss(y, tx, w)
        w = w - gamma * grad

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Computes the weights and the losses by applying the logistic regression algorithm.

    :param y: expected labels vector
    :param tx: data matrix
    :param lambda_: the lambda factor to apply to the weights
    :param initial_w: the initial weights to start with
    :param max_iters: the maximal number of iterations to apply
    :param gamma: the gamma factor to apply to the gradient
    :returns: w - computed weights
              loss - computed loss
    """

    loss = math.inf
    w = initial_w
    size = len(w)

    for i in range(max_iters):
        grad = calculate_gradient(y, tx, w) + (2 * lambda_ * w) / size

        loss = calculate_loss(y, tx, w) + (lambda_ * np.matmul(w.T, w)) / size
        w = w - gamma * grad

    return w, loss
