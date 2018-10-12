import numpy as np

def standardize(x):
    centered = x - np.mean(x, axis=0)
    normed = centered / np.std(centered, axis=0)

    return normed
