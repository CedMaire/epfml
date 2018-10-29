import numpy as np
from data_processor import build_model_data

"""
Functions related to data loading from the disk.
"""

# All the file pathes.
DATA_PATH_SAMPLE_SUBMISSION = "../data/sample-submission.csv"
DATA_PATH_SAMPLE_SUBMISSION_TEST = "../data/sample-submission-test.csv"
DATA_PATH_TEST = "../data/test.csv"
DATA_PATH_TRAIN = "../data/train.csv"

def load_data(data_path):
    """
    Loads the data set from disk and returns a model (triplet) formed from the labels, the feature matrix and the sample ids.
    It returns the data set splitted into multiple subsets. Each element of the returned triplet is an array of vectors or
    matrices depending on the expected type.

    (Function greatly inspired from the given file "proj1_helpers.py" given by the professor.)

    :param data_path: path of the CSV file to load
    :returns: ys - array of vectors of labels
              txs - array of matrices of the data features (rows are samples, columns are features)
              ids - array of vectors of ids
    """

    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)

    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1
#    yb[np.where(y == "b")] = 0

    ys, txs, ids = build_model_data(yb, input_data, ids)

    return ys, txs, ids
