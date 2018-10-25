import numpy as np

def calculate_mse(e):
    return ((1/2) * np.mean(e ** 2))

def calculate_mae(e):
    return np.mean(np.abs(e))

def compute_loss(y, tx, w, mse=True):
    err = y - tx.dot(w)

    if mse:
        return calculate_mse(err)
    else:
        return calculate_mae(err)
