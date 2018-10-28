import numpy as np

def logistic_function(z):
    return 1.0 / (1 + np.exp(-z))

def calculate_loss(y, tx, w):
    #calculate the sigma
    sigma = logistic_function(tx.dot(w))
    #compute the loss for the two classes
    loss_1 = -np.matmul(y.T,(np.log(sigma)))
    loss_2 = np.matmul((1 - y).T,(np.log(1 - sigma)))
    
    return loss_1 + loss_2 / len(sigma)

def calculate_gradient(y, tx, w):
    #calculate the sigma
    sigma = logistic_function(np.matmul(tx,w))

    z = sigma - y
    grad = np.matmul(tx.T,z)

    return grad / (len(sigma)/2)

def logistic_regression_gradient_descent(y, tx, gamma, max_iter):
    # initinalize parameters
    losses = []
    w = np.zeros((tx.shape[1], 1))
    ws = [w]
    y = np.array([y])
    
    for iter in range(max_iter):
        grad = calculate_gradient(y, tx, w)
        loss = calculate_loss(y, tx, w)
        w = w - gamma * grad
        losses.append(loss)
        ws.append(w)

    return ws[len(ws) - 1].T[0], losses[len(losses) - 1][0][0]
            

def regularized_logistic_regression_gradient_descent(y, tx, gamma, max_iter, lambd):
    # initialize parameters
    losses = []
    w = np.zeros((tx.shape[1], 1))
    ws = [w]
    y = np.array([y])
    size = len(w)
    for iter in range(max_iter):
        grad = calculate_gradient(y, tx, w) + (2 * lambd * w)/ size
        loss = calculate_loss(y, tx, w) + (lambd * np.matmul(w.T, w))/ size
        w = w - gamma * grad
        losses.append(loss)
        ws.append(w)
        
    return ws[len(ws) - 1].T[0], losses[len(losses) - 1][0][0]
