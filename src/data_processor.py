import numpy as np

def standardize(x):
    centered = x - np.mean(x, axis=0)
    normed = centered / np.std(centered, axis=0)

    return normed

def build_model_data(y, x):
    x = standardize(x)

    return y, np.c_[np.ones(len(y)), x]
