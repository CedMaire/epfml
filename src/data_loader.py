import os
import numpy as np
import matplotlib.image as mpimg

"""
Functions related to data loading from the disk.
"""

ROOT_DIR = "data/training/"
IMAGE_DIR = ROOT_DIR + "images/"
GROUNDTRUTH_DIR = ROOT_DIR + "groundtruth/"

def load_image(file_):
    """
    Loads a file as a MathPlotLib image object.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param file_: the file path
    :returns: mpimg - the image as a MathPlotLib image object
    """

    return mpimg.imread(file_)

def load_images(max_to_load=None):
    """
    Loads all images and their groundtruth version from the default folder.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param max_to_load: the maximum number of images to load
    :returns: images - the images
              images_groundtruth - the groundtruth versions of images
    """

    files = os.listdir(IMAGE_DIR)

    if max_to_load == None:
        max_to_load = len(files)

    print("Loading " + str(max_to_load) + " images...")
    images = [load_image(IMAGE_DIR + files[i]) for i in range(max_to_load)]
    print("Loading " + str(max_to_load) + " groundtruth images...")
    images_groundtruth = [load_image(GROUNDTRUTH_DIR + files[i]) for i in range(max_to_load)]

    return images, images_groundtruth



























#from data_processor import build_model_data
#
## All the file pathes.
#DATA_PATH_SAMPLE_SUBMISSION = "../data/sample-submission.csv"
#DATA_PATH_SAMPLE_SUBMISSION_TEST = "../data/sample-submission-test.csv"
#DATA_PATH_TEST = "../data/test.csv"
#DATA_PATH_TRAIN = "../data/train.csv"
#
#def load_data(data_path):
#    """
#    Loads the data set from disk and returns a model (triplet) formed from the labels, the feature matrix and the sample ids.
#    It returns the data set splitted into multiple subsets. Each element of the returned triplet is an array of vectors or
#    matrices depending on the expected type.
#
#    (Function greatly inspired from the given file "proj1_helpers.py" given by the professor.)
#
#    :param data_path: path of the CSV file to load
#    :returns: ys - array of vectors of labels
#              txs - array of matrices of the data features (rows are samples, columns are features)
#              ids - array of vectors of ids
#    """
#
#    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
#    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
#
#    ids = x[:, 0].astype(np.int)
#    input_data = x[:, 2:]
#
#    yb = np.ones(len(y))
#    yb[np.where(y == "b")] = -1
#
#    ys, txs, ids = build_model_data(yb, input_data, ids)
#
#    return ys, txs, ids
