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
