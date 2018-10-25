import numpy as np

"""
Data Cleaning:
    * Remove unwanted, duplicate or irrelevant results
    * Filter unwanted outliers
    * Handle Missing data? (Replace -999 by the mean of the remaining?)
"""

UNDEFINED_VALUE = -999

def remove_undefined_columns(x):
    bool_table = (x == UNDEFINED_VALUE)
    idx = bool_table.any(axis=0)

    return x[:, ~idx]

def standardize(x):
    centered = x - np.mean(x, axis=0)
    normed = centered / np.std(centered, axis=0)

    return normed

def build_model_data(y, x):
    x = remove_undefined_columns(x)
    x = standardize(x)

    return y, np.c_[np.ones(len(y)), x]

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))

        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
