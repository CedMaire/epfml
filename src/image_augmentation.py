#!/usr/bin/python

import imgaug as ia
import data_processor
from imgaug import augmenters as iaa
import imageio
import numpy as np
import os
import re
import matplotlib.image as mpimg
from PIL import Image

ia.seed(2018)

ROOT_DIR = "data/training/"
IMAGE_DIR = ROOT_DIR + "images/"
IMAGE_TEST_DIR = "data/test/"

GROUNDTRUTH_DIR = ROOT_DIR + "groundtruth/"

STRING_SAT_IMAGE_ = "satImage_"
STRING_PNG_EXT = ".png"
STRING_TEST_ = "test_"

AUGMENTATIONS = [iaa.Affine(rotate=(-180, 180)),
                 iaa.Fliplr(0.33),
                 iaa.Flipud(0.33),
                 iaa.PiecewiseAffine(scale=(0.001, 0.02))]

def natural_key(string_):
    """
    Natural key comparator to correctly sort numbers as a string. This way "2" actually comes before "12", which is not
    the case when strings are used. It generates an array by splitting the string at each group of digits.

    Inspired from: https://stackoverflow.com/a/3033342

    :param string_: string that will be compared
    :return array: string splitted by group of digits
    """

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def save_images(images, images_groundtruth):
    """
    Saves the images on the disk.

    :param images: array of images
    :param images_groundtruth: array of groundtruth images
    """

    for file_ in [file_ for file_ in os.listdir(IMAGE_DIR)]:
        os.remove(os.path.join(IMAGE_DIR, file_))

    for index, image in enumerate(images):
        filename = (IMAGE_DIR + STRING_SAT_IMAGE_ + "%.4d" + STRING_PNG_EXT) % (index + 1)
        Image.fromarray(np.uint8(image)).convert("RGB").save(filename)

    for file_ in [file_ for file_ in os.listdir(GROUNDTRUTH_DIR)]:
        os.remove(os.path.join(GROUNDTRUTH_DIR, file_))

    for index, image_groundtruth in enumerate(images_groundtruth):
        filename = (GROUNDTRUTH_DIR + STRING_SAT_IMAGE_ + "%.4d" + STRING_PNG_EXT) % (index + 1)
        Image.fromarray(np.uint8(image_groundtruth * 255)).convert("L").save(filename)

def load_images():
    """
    Loads the images from the disk.

    :return array: array of images
    :return array: array of groundtruth images
    """

    files = sorted(os.listdir(IMAGE_DIR), key=natural_key)

    return [np.array(Image.fromarray(np.uint8(mpimg.imread(IMAGE_DIR + files[i]) * 255)).convert("RGB"), dtype=np.uint8) for i in range(len(files))], [np.array(Image.fromarray(np.uint8(mpimg.imread(GROUNDTRUTH_DIR + files[i]) * 255)).convert("I"), dtype=np.bool_) for i in range(len(files))]

def augment_data(images, images_groundtruth, multiplier):
    """
    Augments the images by applying a range of transformation randomly.

    :param images: array of images
    :param images_groundtruth: array of groundtruth images
    :param multiplier: the number of times each image should be augmented randomly
    :return array: array of all original and augmented images
    :return array: array of all original and augmented groundtruth images
    """

    images_groundtruth = list(map(lambda img: ia.SegmentationMapOnImage(img, shape=images[0].shape, nb_classes=2), images_groundtruth))

    images_augmented, images_groundtruth_augmented = [], []

    images_augmented.extend(images)
    images_groundtruth_augmented.extend(images_groundtruth)

    for i in range(multiplier):
        print("Multiplying Data: " + str(i))
        for image, image_groundtruth in zip(images, images_groundtruth):
            sequence = iaa.SomeOf(np.random.randint(1, 4), AUGMENTATIONS, random_order=True).to_deterministic()

            images_augmented.append(sequence.augment_image(image))
            images_groundtruth_augmented.append(sequence.augment_segmentation_maps([image_groundtruth])[0])

    return np.array(images_augmented), np.array(list(map(lambda img: np.asarray(img.get_arr_int(), dtype=np.float32), images_groundtruth_augmented)))

def get_augmented_data(multiplier):
    """
    Loads the original images, augments and returns them.

    :param multiplier: the number of times each image should be augmented randomly
    :return array: array of all original and augmented images
    :return array: array of all original and augmented groundtruth images
    """

    images, images_groundtruth = load_images()

    return augment_data(images[:100], images_groundtruth[:100], multiplier)

if __name__ == "__main__":
    images_augmented, images_groundtruth_augmented = get_augmented_data(14)
    save_images(images_augmented, images_groundtruth_augmented)
