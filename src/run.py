"""
It will load our best model and run it on the test set.
BUT it will not train it, if you want to run the full process, please run train.py

Benjamin Délèze, Antonio Morais, Cedric Maire
"""

import matplotlib.image as mpimg
from matplotlib import cm

from PIL import Image

import keras
from keras.models import load_model
from mask_to_submission import masks_to_submission
from post_processing_helper import suppress_single_roads
from post_processing_helper import add_road_when_line_of_roads
from post_processing_helper import suppress_group_roads_surrounded
from data_loader_saver_helper import label_to_img
from data_loader_saver_helper import img_crop

import tensorflow.python.platform

import numpy as np

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

if __name__ == '__main__':

    #Load the model
    road_model = load_model('road_model_20E15002.h5py')
    road_model.summary()  
    
    TEST_IMAGE_DIR = "data/test_set_images/images/"
    submission_filename = 'data/test_submission.csv'
    image_filenames = []
    for i in range(1, 51):
        img = mpimg.imread(TEST_IMAGE_DIR + "test_"+ '%.1d' % i + "/" + "test_"+ '%.1d' % i + ".png")
        IMG_WIDTH = img.shape[0]
        IMG_HEIGHT = img.shape[1]
        N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)
        img_patches = img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
        data_to_test = np.asarray(img_patches)
        predicted_image = road_model.predict(data_to_test)
        
        #Post-processing
        size_width = int(IMG_WIDTH/IMG_PATCH_SIZE)
        size_height = int(IMG_HEIGHT/IMG_PATCH_SIZE)
        predicted_image = suppress_single_roads(predicted_image, size_width, size_height)
        predicted_image = add_road_when_line_of_roads(predicted_image, size_width, size_height, 3)
        for size_square in range(2,4):
            predicted_image = suppress_group_roads_surrounded(predicted_image, size_width, size_height, size_square)
        predicted_image = label_to_img(IMG_WIDTH, IMG_HEIGHT, IMG_PATCH_SIZE, IMG_PATCH_SIZE, predicted_image)
        
        #Save image
        image_filename = 'data/test_set_images/groundtruth/satImage_' + '%.3d' % i + '.png'
        imag = Image.fromarray(np.uint8(cm.gist_earth(predicted_image)*255))
        imag.save(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
