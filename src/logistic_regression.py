import datetime
import numpy as np

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    sigma = sigmoid(tx.dot(w))
    return -(y.T.dot(np.log(sigma)) + (1 - y).T.dot(np.log(1 - sigma))) / len(sigma)

def calculate_gradient(y, tx, w):
    sigma = sigmoid(tx.dot(w))
    z = sigma - y
    grad = tx.T.dot(z)
    return grad

def logistic_regression_gradient_descent(y, tx, gamma, max_iter):
    # init parameters
    losses = []
    # initialize
    w = np.zeros((tx.shape[1], 1))
    ws = [w]
    y = np.array([y])
    y = y.T
    for iter in range(max_iter):
        grad = calculate_gradient(y, tx, w)
        loss = calculate_loss(y, tx, w)
        w = w - gamma * grad
        losses.append(loss)
        ws.append(w)
        if(iter%100 == 0):
            pass
#            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    return ws[len(ws) - 1].T[0], losses[len(losses) - 1][0][0]
            
def regularized_logistic_regression(y, tx, w, lambd):
    num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + lambd * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambd * w
    return loss, gradient

def regularized_logistic_regression_gradient_descent(y, tx, gamma, max_iter, lambd):
    # init parameters
    losses = []
    # initialize
    w = np.zeros((tx.shape[1], 1))
    ws = [w]
    y = np.array([y])
    y = y.T
    for iter in range(max_iter):
        loss, grad = regularized_logistic_regression(y, tx, w, lambd)
        w = w - gamma * grad
        losses.append(loss)
        ws.append(w)
        if(iter%100 == 0):
            pass
#            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    return ws[len(ws) - 1].T[0], losses[len(losses) - 1][0][0]
