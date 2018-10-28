import numpy as np

"""
Functions related to the prediction of the labels.
"""

def predict_labels(weights, data):
    """
    Predicts the labels given a data set and the pre-computed weights.

    (Function greatly inspired from the given file "proj1_helpers.py" given by the professor.)

    :param weights: pre-computed weights
    :param data: the data set
    :returns: a vector of predictions
    """

    y_pred = np.dot(data, weights)

    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred
