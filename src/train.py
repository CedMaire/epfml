"""
This file will train will reproduce the model we used to reach our best result.
In order to run it, you need to get the additional data (1500 images)
The training can take several hours

Benjamin Délèze, Cedric Maire, Antonio Morais
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image

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
from cnn_helper import balance_data
from cnn_helper import create_model

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

TRAINING_SIZE = 1500
batch_size = 64
epochs = 150
num_classes = 2
should_load_model = False

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

if __name__ == '__main__':   

    if not should_load_model:
        data_dir = 'data/training/'
        train_data_filename = data_dir + 'images/'
        train_labels_filename = data_dir + 'groundtruth/'

        # Extract it into numpy arrays.
        #train_data = extract_data(train_data_filename, TRAINING_SIZE)
        train_data_unbalanced = extract_data(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE)
        train_labels_unbalanced = extract_labels(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE)
    
        #balance the data to get the same number of each label
        train_data, train_labels = balance_data(train_data_unbalanced, train_labels_unbalanced)
        
        print(train_data.shape)
        print(train_labels.shape)
    
        #split our data set in three: training set, validation set and testing set
        step_train_X, test_X, step_train_label, test_label = train_test_split(train_data, train_labels, test_size=0.1, random_state=13)
        train_X, validation_X, train_label, validation_label = train_test_split(step_train_X, step_train_label, test_size=0.1, random_state=13)
        
        #create the model
        road_model = create_model(num_classes)
        
        #fit the data
        road_train_dropout = road_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(validation_X, validation_label))

        road_model.save("road_model_sigmoid150.h5py")
        
        #Plot informations about the training
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
        road_model = load_model('road_model_test.h5py')
        road_model.summary()  
    
    test_eval = road_model.evaluate(test_X, test_label, verbose=0)   
    
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    
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
