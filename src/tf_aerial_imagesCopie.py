"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image

import code

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from sklearn.model_selection import train_test_split
from mask_to_submission import masks_to_submission
from post_processing_helper import suppress_single_roads
from post_processing_helper import add_road_when_line_of_roads
from post_processing_helper import suppress_group_roads_surrounded
from data_loader_saver_helper import extract_data
from data_loader_saver_helper import extract_labels
from data_loader_saver_helper import label_to_img
from data_loader_saver_helper import img_crop

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

TRAINING_SIZE = 10

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

if __name__ == '__main__':
    
    data_dir = 'data/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    #train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_data = extract_data(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE)
   
    c0 = 0  # bgrd
    c1 = 0  # road
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(train_data.shape)
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
        
    print(train_data.shape)
    print(train_labels.shape)
    step_train_X, test_X, step_train_label, test_label = train_test_split(train_data, train_labels, test_size=0.1, random_state=13)
    train_X, validation_X, train_label, validation_label = train_test_split(step_train_X, step_train_label, test_size=0.1, random_state=13)
   
    should_load_model = False
    if not should_load_model:
        batch_size = 125
        epochs = 10
        num_classes = 2

        road_model = Sequential()
        road_model.add(Conv2D(64, kernel_size=(5, 5),activation='linear',input_shape=(16,16,3),padding='same'))
        road_model.add(LeakyReLU(alpha=0.1))
        road_model.add(MaxPooling2D((2, 2),padding='same'))
        road_model.add(Dropout(0.25))
        road_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        road_model.add(LeakyReLU(alpha=0.1))
        road_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        road_model.add(Dropout(0.25))
        road_model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
        road_model.add(LeakyReLU(alpha=0.1))                  
        road_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        road_model.add(Dropout(0.25))
        road_model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
        road_model.add(LeakyReLU(alpha=0.1))                  
        road_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        road_model.add(Dropout(0.25))
        road_model.add(Flatten())
        road_model.add(Dense(128, activation='linear'))
        road_model.add(LeakyReLU(alpha=0.1))   
        road_model.add(Dropout(0.5))
        road_model.add(Dense(num_classes, activation='softmax'))

        road_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

        road_model.summary()

        road_train_dropout = road_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(validation_X, validation_label))

        road_model.save("road_model_test2.h5py")
        
        accuracy = road_train_dropout.history['acc']
        val_accuracy = road_train_dropout.history['val_acc']
        loss = road_train_dropout.history['loss']
        val_loss = road_train_dropout.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
    else : 
        road_model = load_model('road_model_20E15002.h5py')
        road_model.summary()  
    
    test_eval = road_model.evaluate(test_X, test_label, verbose=0)   
    
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    
    TEST_IMAGE_DIR = "data/test_set_images/images/"
    submission_filename = 'data/test_submission.csv'
    image_filenames = []
    for i in range(1, 51):
        img = mpimg.imread(TEST_IMAGE_DIR + "test_"+ '%.1d' % i + "/" + "test_"+ '%.1d' % i + ".png")
        #img = mpimg.imread(train_data_filename + "satImage_"+ '%.3d' % i + ".png")
        IMG_WIDTH = img.shape[0]
        IMG_HEIGHT = img.shape[1]
        N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)
        img_patches = img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
        data_to_test = np.asarray(img_patches)
        predicted_image = road_model.predict(data_to_test)
        size_width = int(IMG_WIDTH/IMG_PATCH_SIZE)
        size_height = int(IMG_HEIGHT/IMG_PATCH_SIZE)
        predicted_image = suppress_single_roads(predicted_image, size_width, size_height)
        predicted_image = add_road_when_line_of_roads(predicted_image, size_width, size_height, 3)
        for size_square in range(2,4):
            predicted_image = suppress_group_roads_surrounded(predicted_image, size_width, size_height, size_square)
        predicted_image = label_to_img(IMG_WIDTH, IMG_HEIGHT, IMG_PATCH_SIZE, IMG_PATCH_SIZE, predicted_image)
        image_filename = 'data/test_set_images/groundtruth/satImage_' + '%.3d' % i + '.png'
        imag = Image.fromarray(np.uint8(cm.gist_earth(predicted_image)*255))
        imag.save(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
