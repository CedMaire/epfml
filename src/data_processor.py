import numpy as np
import matplotlib.pyplot as plt
import data_loader

"""
Functions related to data processing.
"""

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
    X = np.asarray([extract_features_mean_var_2d(patch) for patch in patches_images])
    Y = np.asarray([value_to_class(np.mean(patch)) for patch in patches_groundtruth])
    print("Computed " + str(X.shape[0]) + " features.")
    print("Feature dimension = " + str(X.shape[1]))
    print("Number of classes = " + str(np.max(Y)))

    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print("Class 0: " + str(len(Y0)) + " samples")
    print("Class 1: " + str(len(Y1)) + " samples")

    show_image(patches_groundtruth[Y1[3]])
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.show()

images, images_groundtruth = data_loader.load_images()
patches_images, patches_groundtruth = generate_patches(images, images_groundtruth)
build_model(patches_images, patches_groundtruth)



























# Value representing the fact that data is undefined or not relevant.
UNDEFINED_VALUE = -999

# All the different interesting columns.
col_DER_mass_MMC = 2
col_PRI_jet_num = 24
col_set_less_1 = np.asarray([6, 7, 8, 14, 28, 29, 30]) # All the columns undefined if PRI_jet_num <= 1
col_set_eq_0 = np.asarray([25, 26, 27]) # All the columns undefined if PRI_jet_num == 0

def split_data_meaningfuly(y, x, ids):
    """
    Splits the data into 8 subsets depending on the values in columns DER_mass_MMC and PRI_jet_num.
    DER_mass_MMC can be undefined without any reason, so we split the data into two set depending on the fact that this
        column is defined or not.
    PRI_jet_num can only contain 4 different values (0, 1, 2, 3): we split the two data subsets into 4 depending on these
        4 values.
    Once we have the 8 subsets we can begin cleaning the data by removing specific columns that will contain the exact
        same value throughout the whole set (DER_mass_MMC and PRI_jet_num) or that will be undefined because of the
        in PRI_jet_num (see the official PDF given by the Kaggle [http://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf]).
    The returned triplet form 8 subsets of the input data, the 3 arrays are in the same order, i.e. ys[i], txs[i] and ids[i]
        form one specific subset.

    :param y: vector containing the expected labels
    :param x: data matrix (rows are samples, columns are features)
    :param ids: vector containing the ids of the samples
    :returns: ys - array of vectors of labels
              txs - array of matrices of the data features (rows are samples, columns are features)
              ids - array of vectors of ids
    """

    # First we add the column of ids and labels so that we get back the correct number of colums.
    x = np.c_[ids, y, x]

    # Combine the interesting columns.
    col_set_DER_PRI = np.sort(np.asarray([col_DER_mass_MMC, col_PRI_jet_num]))
    col_set_all = np.sort(np.append(np.append(col_set_DER_PRI, col_set_less_1), col_set_eq_0))
    col_set_DER_PRI_less_1 = np.sort(np.append(col_set_DER_PRI, col_set_less_1))

    # First split depending on the fact that DER_mass_MMC is defined or not.
    DER_MASS_MMC_undef = x[x[:, col_DER_mass_MMC] == UNDEFINED_VALUE]
    DER_MASS_MMC_def = x[x[:, col_DER_mass_MMC] != UNDEFINED_VALUE]

    # Second split depending on the four values of PRI_jet_num.
    DER_MASS_MMC_undef_PRI_jet_num_0 = DER_MASS_MMC_undef[DER_MASS_MMC_undef[:, col_PRI_jet_num] == 0]
    DER_MASS_MMC_undef_PRI_jet_num_1 = DER_MASS_MMC_undef[DER_MASS_MMC_undef[:, col_PRI_jet_num] == 1]
    DER_MASS_MMC_undef_PRI_jet_num_2 = DER_MASS_MMC_undef[DER_MASS_MMC_undef[:, col_PRI_jet_num] == 2]
    DER_MASS_MMC_undef_PRI_jet_num_3 = DER_MASS_MMC_undef[DER_MASS_MMC_undef[:, col_PRI_jet_num] == 3]

    DER_MASS_MMC_def_PRI_jet_num_0 = DER_MASS_MMC_def[DER_MASS_MMC_def[:, col_PRI_jet_num] == 0]
    DER_MASS_MMC_def_PRI_jet_num_1 = DER_MASS_MMC_def[DER_MASS_MMC_def[:, col_PRI_jet_num] == 1]
    DER_MASS_MMC_def_PRI_jet_num_2 = DER_MASS_MMC_def[DER_MASS_MMC_def[:, col_PRI_jet_num] == 2]
    DER_MASS_MMC_def_PRI_jet_num_3 = DER_MASS_MMC_def[DER_MASS_MMC_def[:, col_PRI_jet_num] == 3]

    # Remove all the columns that are now useless since they are the same for each set of data or undefined.
    DER_MASS_MMC_undef_PRI_jet_num_0 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_0, col_set_all, axis=1)
    DER_MASS_MMC_undef_PRI_jet_num_1 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_1, col_set_DER_PRI_less_1, axis=1)
    DER_MASS_MMC_undef_PRI_jet_num_2 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_2, col_set_DER_PRI, axis=1)
    DER_MASS_MMC_undef_PRI_jet_num_3 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_3, col_set_DER_PRI, axis=1)

    DER_MASS_MMC_def_PRI_jet_num_0 = np.delete(DER_MASS_MMC_def_PRI_jet_num_0, col_set_all, axis=1)
    DER_MASS_MMC_def_PRI_jet_num_1 = np.delete(DER_MASS_MMC_def_PRI_jet_num_1, col_set_DER_PRI_less_1, axis=1)
    DER_MASS_MMC_def_PRI_jet_num_2 = np.delete(DER_MASS_MMC_def_PRI_jet_num_2, col_set_DER_PRI, axis=1)
    DER_MASS_MMC_def_PRI_jet_num_3 = np.delete(DER_MASS_MMC_def_PRI_jet_num_3, col_set_DER_PRI, axis=1)

    # Creation of the output arrays.
    ys = np.asarray([DER_MASS_MMC_undef_PRI_jet_num_0[:,1],
                        DER_MASS_MMC_undef_PRI_jet_num_1[:,1],
                        DER_MASS_MMC_undef_PRI_jet_num_2[:,1],
                        DER_MASS_MMC_undef_PRI_jet_num_3[:,1],
                        DER_MASS_MMC_def_PRI_jet_num_0[:,1],
                        DER_MASS_MMC_def_PRI_jet_num_1[:,1],
                        DER_MASS_MMC_def_PRI_jet_num_2[:,1],
                        DER_MASS_MMC_def_PRI_jet_num_3[:,1]])
    txs = np.asarray([DER_MASS_MMC_undef_PRI_jet_num_0[:,2:-1], # Last column is constant 0.
                        DER_MASS_MMC_undef_PRI_jet_num_1[:,2:],
                        DER_MASS_MMC_undef_PRI_jet_num_2[:,2:],
                        DER_MASS_MMC_undef_PRI_jet_num_3[:,2:],
                        DER_MASS_MMC_def_PRI_jet_num_0[:,2:-1], # Last column is constant 0.
                        DER_MASS_MMC_def_PRI_jet_num_1[:,2:],
                        DER_MASS_MMC_def_PRI_jet_num_2[:,2:],
                        DER_MASS_MMC_def_PRI_jet_num_3[:,2:]])
    ids = np.asarray([DER_MASS_MMC_undef_PRI_jet_num_0[:,0],
                        DER_MASS_MMC_undef_PRI_jet_num_1[:,0],
                        DER_MASS_MMC_undef_PRI_jet_num_2[:,0],
                        DER_MASS_MMC_undef_PRI_jet_num_3[:,0],
                        DER_MASS_MMC_def_PRI_jet_num_0[:,0],
                        DER_MASS_MMC_def_PRI_jet_num_1[:,0],
                        DER_MASS_MMC_def_PRI_jet_num_2[:,0],
                        DER_MASS_MMC_def_PRI_jet_num_3[:,0]])

    return ys, txs, ids

def standardize(x):
    """
    Standardizes a data set. For each column it substract the mean of this feature and then devides each column by the
    standard variation of this column (everything is done element-wise).

    :param x: matrix of data
    :returns: the standardized matrix
    """

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    x = x - x_mean
    return x / x_std

def build_model_data(y, x, ids):
    """
    Builds a model of data to be used for machine learning algorithms. Instead of creatung one model this method returns
    8 different models. A model is composed of a triplet: the expected labels (y), the matrix of samples and features (tx)
    and the id of the samples (ids). This method return 8 complete models, so ys is a list of expected labels, txs a list
    of matrices of samples and features and ids a list of id. yys, txs and ids all share the same ordering, this means that
    a same index forms one data model.

    :param y: vector of expected labels
    :param x: matrix of data (rows are samples, columns are features)
    :param ids: vector of sample ids
    :returns: ys - array of vectors of labels
              txs - array of matrices of the data features (rows are samples, columns are features)
              ids - array of vectors of ids
    """

    ys, txs, ids = split_data_meaningfuly(y, x, ids)
    txs = np.asarray(list(map(lambda tx: np.c_[np.ones(len(tx)), standardize(tx)], txs)))

    return ys, txs, ids
