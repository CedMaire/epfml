import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import data_loader
from mask_to_submission import masks_to_submission
from learn import model_linear_logistic_regression
from PIL import Image
from skimage import color
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

TEST_IMAGE_DIR = "data/test_set_images/images/"
"""
Functions related to data processing.
"""

def standardize(x):
    """
    Standardizes a data set. For each column it substract the mean of this feature and then devides each column by the
    standard variation of this column (everything is done element-wise).

    :param x: matrix of data
    :returns: the standardized matrix
    """

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    return (x - x_mean) / x_std

def show_image(image):
    """
    Show the image on the screen.

    :param image: the image to show
    """

    plt.imshow(image, cmap="Greys_r")
    plt.show()

def image_float_to_uint8(image):
    """
    Converts the pixels of an image from floats to unsigned bytes.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param image: the image to convert
    :returns: image - the converted image where each pixel is an unsigned bytes
    """

    rimg = image - np.min(image)

    return (rimg / np.max(rimg) * 255).round().astype(np.uint8)

def concatenate_images(image, image_groundtruth):
    """
    Concatenates the original image with its groundtruth image to have both side by side.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param image: the original image
    :param image_groundtruth: the image_groundtruth image
    :returns: image_concatenated - the concatenation of both images
    """

    number_channels = len(image_groundtruth.shape)

    width = image_groundtruth.shape[0]
    height = image_groundtruth.shape[1]

    if number_channels == 3:
        image_concatenated = np.concatenate([image, image_groundtruth], axis=1)
    else:
        image_groundtruth_uint8 = image_float_to_uint8(image_groundtruth)
        image_uint8 = image_float_to_uint8(image)

        image_groundtruth_3channels = np.zeros((width, height, 3), dtype=np.uint8)
        image_groundtruth_3channels[:,:,0] = image_groundtruth_uint8
        image_groundtruth_3channels[:,:,1] = image_groundtruth_uint8
        image_groundtruth_3channels[:,:,2] = image_groundtruth_uint8

        image_concatenated = np.concatenate([image_uint8, image_groundtruth_3channels], axis=1)

    return image_concatenated

def crop_image(image, width, height):
    """
    Crops the image and returns all the generated patches of the given dimensions.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param image: the original image
    :param width: the wanted width
    :param height: the wanted height
    :returns: patches - all the patches of the given dimensions
    """

    patches = []
    for i in range(0, image.shape[1], height):
        for j in range(0, image.shape[0], width):
            if len(image.shape) < 3:
                patches.append(image[j:j+width, i:i+height])
            else:
                patches.append(image[j:j+width, i:i+height,:])

    return patches

def generate_patches(images, images_groundtruth, patch_size=16):
    """
    Generates patches of size patch_size given as parameter.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param images: the images to get the patches from
    :param images_groundtruth: the groundtruth images to get the patches from
    :param patch_size: the size of a patch, defaults to 16
    :returns: patches_images - generated patches from images
              patches_groundtruth - generated patches from images_groundtruth
    """

    patches_images = [crop_image(image, patch_size, patch_size) for image in images]
    patches_groundtruth = [crop_image(image_groundtruth, patch_size, patch_size) for image_groundtruth in images_groundtruth]

    patches_images = np.asarray([patch for patches in patches_images for patch in patches])
    patches_groundtruth =  np.asarray([patch for patches in patches_groundtruth for patch in patches])

    return patches_images, patches_groundtruth

def extract_features_mean_var_6d(image):
    """
    Extracts 6-dimensional features for the mean and variance of an image/patch.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param image: the images to get the features from
    :returns: features - generated features from image/patch
    """

    return np.append(np.mean(image, axis=(0,1)), np.var(image, axis=(0,1)))

def extract_features_mean_var_12d(image):
    """
    Extracts 6-dimensional features for the mean and variance of an image/patch.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param image: the images to get the features from
    :returns: features - generated features from image/patch
    """

    return np.append(np.append(np.append(np.mean(image, axis=(0,1)), np.var(image, axis=(0,1))), np.mean(color.rgb2gray(image), axis=(0,1))), np.var(color.rgb2gray(image), axis=(0,1)))


def extract_features_mean_var_2d(image):
    """
    Extracts 2-dimensional features for the mean and variance of an image/patch.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param image: the images to get the features from
    :returns: features - generated features from image/patch
    """

    return np.append(np.mean(image), np.var(image))

def value_to_class(values, foreground_threshold=0.25):
    """
    Converts values to a class.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param values: the values to compute the class from
    :param foreground_threshold: percentage of pixels > 1 required to assign a foreground label to a patch
    :returns: class - 0 or 1 as a class label
    """

    return 1 if np.sum(values) > foreground_threshold else 0

def build_model(patches_images, patches_groundtruth):
    """
    Builds a model from the given patches.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param patches_images: the patches from orginial images
    :param patches_groundtruth: the patches from groundtruth images
    :returns: Y - the labels vector
              X - the features matrix
    """

    X = np.asarray([extract_features_mean_var_6d(patch) for patch in patches_images])
    Y = np.asarray([value_to_class(np.mean(patch)) for patch in patches_groundtruth])
    print("Computed " + str(X.shape[0]) + " features.")
    print("Feature dimension = " + str(X.shape[1]))
    print("Number of classes = " + str(np.max(Y)))

    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print("Class 0: " + str(len(Y0)) + " samples")
    print("Class 1: " + str(len(Y1)) + " samples")
    """
    show_image(patches_groundtruth[Y1[3]])
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors="k", cmap=plt.cm.Paired)
    plt.show()
    """
    return Y, X

def labels_to_image(image_width, image_height, width, height, labels):
    """
    Converts labels to an image.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param image_width: the wanted image width
    :param image_height: the wanted image height
    :param width: the patch width
    :param height: the patch height
    :param labels: the labels to create the image from
    :returns: image - the created image
    """

    image = np.zeros([image_width, image_height])
    index = 0

    for i in range(0, image_height, height):
        for j in range(0, image_width, width):
            image[j:j+width, i:i+height] = labels[index]
            index = index + 1

    return image

def image_overlay(image, image_predicted):
    """
    Creates an overlay on the original image with the prediction.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param image: the original image
    :param image_predicted: the predicted image
    :returns: overlayed_image - the original image with the prediction overlayed
    """

    width, height = image.shape[0], image.shape[1]

    color_mask = np.zeros([width, height, 3], dtype=np.uint8)
    color_mask[:,:,0] = image_predicted * 255

    image_uint8 = image_float_to_uint8(image)

    background = Image.fromarray(image_uint8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")

    return Image.blend(background, overlay, 0.2)

def extract_image_features(image, patch_size=16):
    """
    Creates an overlay on the original image with the prediction.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param image: the original image
    :param image_predicted: the predicted image
    :returns: overlayed_image - the original image with the prediction overlayed
    """

    image_patches = crop_image(image, patch_size, patch_size)

    return np.asarray([extract_features_mean_var_6d(patch) for patch in image_patches])

def create_test_submission(ridge_regression):
    submission_filename = 'data/test_submission.csv'
    image_filenames = []
    for i in range(1, 51):
        test_image = data_loader.load_image(TEST_IMAGE_DIR + "test_"+ '%.1d' % i + "/" + "test_"+ '%.1d' % i + ".png")
        Xi = extract_image_features(test_image)
        poly = PolynomialFeatures(4)
        Xi = poly.fit_transform(Xi)
        Zi = ridge_regression.predict(Xi)
        width, height = test_image[i].shape[0], test_image.shape[1]
        predicted_image = labels_to_image(width, height, patch_size, patch_size, Zi)
        image_filename = 'data/test_set_images/groundtruth/satImage_' + '%.3d' % i + '.png'
        #mpimg.imsave(image_filename, test_image)
        imag = Image.fromarray(np.uint8(cm.gist_earth(predicted_image)*255))
        imag.save(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)

if __name__ == "__main__":
    patch_size = 16
    images, images_groundtruth = data_loader.load_images()
    #images_test = data_loader.load_test_images()
    patches_images, patches_groundtruth = generate_patches(images, images_groundtruth, patch_size=patch_size)
    
    #image_index = 2
    #Xi = extract_image_features(images[image_index])
       
    Y, X = build_model(patches_images, patches_groundtruth)
    logistic_regression = linear_model.LogisticRegression(C=1e10, class_weight="balanced")
    #logistic_regression = model_linear_logistic_regression(Y, X) 
    poly = PolynomialFeatures(4)
    X = poly.fit_transform(X)
    
    """
    kf = KFold(n_splits=4)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        logistic_regression = linear_model.LogisticRegression(C=1e10, class_weight="balanced")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        poly = PolynomialFeatures(4)
        X_train = poly.fit_transform(X_train)
        X_test = poly.fit_transform(X_test)
        logistic_regression.fit(X_train, y_train)
        print(f1_score(y_test, logistic_regression.predict(X_test), average='macro')) 
    """
    #scores = cross_val_score(logistic_regression, X, Y, cv=5)
    #print(scores)
    logistic_regression.fit(X, Y)
    create_test_submission(logistic_regression)
    """
    Y, X = build_model(patches_images, patches_groundtruth)
    logistic_regression = model_linear_logistic_regression(Y,X) 
    Xi = extract_image_features(images[1])
    Zi = logistic_regression.predict(Xi)
    print(Zi)
    plt.scatter(Xi[:,0], Xi[:,1], c=Zi, edgecolors="k", cmap=plt.cm.Paired)
    plt.show()

    width, height = images_groundtruth[image_index].shape[0], images_groundtruth[image_index].shape[1]
    predicted_image = labels_to_image(width, height, patch_size, patch_size, Zi)
    concatenated_image = concatenate_images(images[image_index], predicted_image)
    plt.imshow(concatenated_image, cmap='Greys_r')
    plt.show()

    new_image = image_overlay(images[image_index], predicted_image)
    plt.imshow(new_image)
    plt.show()
    """
