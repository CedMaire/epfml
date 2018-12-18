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
#                iaa.ElasticTransformation(alpha=1000, sigma=50),
#                iaa.Dropout([0.05, 0.10]),
                iaa.Fliplr(0.33),
                iaa.Flipud(0.33),
#                iaa.Superpixels(p_replace=0.1, n_segments=200),
#                iaa.Sharpen(alpha=(0.1, 0.6), lightness=(0.7, 1.3)),
                iaa.PiecewiseAffine(scale=(0.001, 0.02))]

def natural_key(string_):
    # https://stackoverflow.com/a/3033342
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def save_images(images, images_groundtruth):
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
    files = sorted(os.listdir(IMAGE_DIR), key=natural_key)

    return [np.array(Image.fromarray(np.uint8(mpimg.imread(IMAGE_DIR + files[i]) * 255)).convert("RGB"), dtype=np.uint8) for i in range(len(files))], [np.array(Image.fromarray(np.uint8(mpimg.imread(GROUNDTRUTH_DIR + files[i]) * 255)).convert("I"), dtype=np.bool_) for i in range(len(files))]

def augment_data(images, images_groundtruth, multiplier):
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
    images, images_groundtruth = load_images()

    return augment_data(images[:100], images_groundtruth[:100], multiplier)

if __name__ == "__main__":
    # https://github.com/aleju/imgaug
    # https://imgaug.readthedocs.io/en/latest/source/examples_segmentation_maps.html

#    print(mpimg.imread("data/training/images/satImage_0190.png").shape)
#    print(mpimg.imread("data/training/groundtruth/satImage_0186.png").shape)

    images_augmented, images_groundtruth_augmented = get_augmented_data(14)
    save_images(images_augmented, images_groundtruth_augmented)

#    data_processor.show_image(data_processor.concatenate_images(images_augmented[0], images_groundtruth_augmented[0]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[10], images_groundtruth_augmented[10]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[100], images_groundtruth_augmented[100]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[110], images_groundtruth_augmented[110]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[120], images_groundtruth_augmented[120]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[130], images_groundtruth_augmented[130]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[140], images_groundtruth_augmented[140]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[150], images_groundtruth_augmented[150]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[160], images_groundtruth_augmented[160]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[170], images_groundtruth_augmented[170]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[180], images_groundtruth_augmented[180]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[190], images_groundtruth_augmented[190]))
#    data_processor.show_image(data_processor.concatenate_images(images_augmented[199], images_groundtruth_augmented[199]))

#    cells = []
#    for image_augmented, image_groundtruth_augmented in zip(images_augmented, images_groundtruth_augmented):
#        cells.append(image_augmented)
#        cells.append(image_groundtruth_augmented.draw_on_image(image_augmented))
#        cells.append(image_groundtruth_augmented.draw(size=image_augmented.shape[:2]))
#
#    imageio.imwrite("test.png", ia.draw_grid(cells, cols=3))
