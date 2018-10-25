import numpy as np
from helpers import compute_mse

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = None
    loss = None

    raise NotImplementedError

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = None
    loss = None

    raise NotImplementedError

    return w, loss

def least_squares(y, tx):

    txTransposed = tx.T
    txDotT = np.dot(txTransposed, tx)
    txTdotY = np.dot(txTransposed, y)
    w = np.linalg.solve(txDotT, txTdotY)

    loss = np.sqrt(2 * compute_mse(y, tx, w))

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
