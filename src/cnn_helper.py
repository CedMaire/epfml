"""
This file contains the functions we used to build our cnn

Benjamin Délèze, Antonio Morais, Cedric Maire
"""
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def balance_data(train_data, train_labels):
    """
    This function will balance the sample such that we get an equal number of samples for each label
    
    Function from the tf_aerial_images.py given on CrowAI
    Credits: Aurelien Lucchi, ETH Zürich
    
    :param train_data: the data to be balanced
    :param train_label: the corresponding labels
    :return train_data: the balanced data
    :return train_label: the balanced labels
    """
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
    return train_data, train_labels

def create_model(num_classes):
    """
    Create and initialize our model such that it is ready to be fit with our data
    
    :param num_classes: the number of classes
    :return road_model: our model
    """
    road_model = Sequential()
    road_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(16,16,3),padding='same'))
    road_model.add(LeakyReLU(alpha=0.1))
    road_model.add(MaxPooling2D((2, 2),padding='same'))
    road_model.add(Dropout(0.25))
    road_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    road_model.add(LeakyReLU(alpha=0.1))
    road_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    road_model.add(Dropout(0.25))
    road_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    road_model.add(LeakyReLU(alpha=0.1))                  
    road_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    road_model.add(Dropout(0.4))
    road_model.add(Flatten())
    road_model.add(Dense(128, activation='linear'))
    road_model.add(LeakyReLU(alpha=0.1))   
    road_model.add(Dropout(0.3))
    road_model.add(Dense(num_classes, activation='sigmoid'))

    road_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    road_model.summary()
    
    return road_model
