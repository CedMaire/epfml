import numpy as np

# Function greatly inspired from the given file "proj1_helpers.py".
def predict_labels(weights, data):
    y_pred = np.dot(data, weights)

    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred
