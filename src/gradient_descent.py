import datetime
import numpy as np
from ipywidgets import IntSlider, interact
from plots import gradient_descent_visualization
from cost_computer import calculate_mse, compute_loss
from data_processor import batch_iter

def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)

    return grad, err

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []

    w = initial_w
    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)

        w = w - gamma * grad

        ws.append(w)
        losses.append(loss)

#        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
#        print("w", w)

    return ws, losses

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []

    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)

            ws.append(w)
            losses.append(loss)

#        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
#        print("w", w)

    return ws, losses

def test_GD(y, tx):
    # Define the parameters of the algorithm.
    max_iters = 100
    gamma = 0.163

    # Initialization
    w_initial = np.array(np.zeros(len(tx[0])))

    # Start gradient descent
    start_time = datetime.datetime.now()
    gradient_ws, gradient_losses = gradient_descent(y, tx, w_initial, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
#    print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

#    # Interact
#    def plot_figure(n_iter):
#        fig = gradient_descent_visualization(
#            gradient_losses, gradient_ws, grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight, n_iter)
#        fig.set_size_inches(10.0, 6.0)
#
#    interact(plot_figure, n_iter=IntSlider(min=1, max=len(gradient_ws)))

    return gradient_ws[len(gradient_ws) - 1], gradient_losses[len(gradient_losses) - 1]

def test_SGD(y, tx):
    # Define the parameters of the algorithm.
    max_iters = 1000
    gamma = 0.01
    batch_size = 1

    # Initialization
    w_initial = np.array(np.zeros(len(tx[0])))

    # Start SGD
    start_time = datetime.datetime.now()
    sgd_ws, sgd_losses = stochastic_gradient_descent(y, tx, w_initial, batch_size, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
#    print("Stochastic Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

#    # Interact
#    def plot_figure(n_iter):
#        fig = gradient_descent_visualization(
#            sgd_losses, sgd_ws, grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight, n_iter)
#        fig.set_size_inches(10.0, 6.0)
#
#    interact(plot_figure, n_iter=IntSlider(min=1, max=len(sgd_ws)))

    return sgd_ws[len(sgd_ws) - 1], sgd_losses[len(sgd_losses) - 1]

'''
gamma = 0.163
Gradient Descent(9999/9999): loss=0.3400115969466962
w [-0.314664    0.02911793 -0.25352996 -0.25510473 -0.02946821  0.03374768
    0.41608333 -0.13089998  0.26794285 -0.00248241  0.00329642 -0.18262544
    0.11492126  0.0123028   0.19170393 -0.00064404 -0.0010565   0.28757911
   -0.00092002  0.00256574  0.12119271  0.00105565 -0.06340221 -0.19522748
   -0.03335308  0.09366113  0.09307171 -0.02713968 -0.02590377 -0.03931533
   -0.10466856]
'''
