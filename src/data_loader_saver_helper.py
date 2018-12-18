"""
This file contains several functions to load/save and extract data/label from images

Benjamin Délèze, Cedric Maire, Antonio Morais
"""

import os
import matplotlib.image as mpimg
import numpy as np

def img_crop(im, w, h):
    """
    Extract patches from a given image
    
    Function from the tf_aerial_images.py given on CrowAI
    Credits: Aurelien Lucchi, ETH Zürich
    
    :param im: the image
    :param w: the width of a patch
    :param h: the height of a patch
    :return list_patches: the list of patches for the image
    """  
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images, IMG_PATCH_SIZE):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    
    Function from the tf_aerial_images.py given on CrowAI
    Credits: Aurelien Lucchi, ETH Zürich
    
    :param filename: the name of the images (string)
    :param num_images: the number of images
    :param IMG_PATCH_SIZE: the size of the patches
    :return: the patches for all images
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.4d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data)

def value_to_class(v):
    """
    Assign a label to a patch v
    
    Function from the tf_aerial_images.py given on CrowAI
    Credits: Aurelien Lucchi, ETH Zürich
    
    :param v: a patch
    :return: the label corresponding to v
    """
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]

def extract_labels(filename, num_images, IMG_PATCH_SIZE):
    """
    Extract the labels into a 1-hot matrix [image index, label index].
    
    Function from the tf_aerial_images.py given on CrowAI
    Credits: Aurelien Lucchi, ETH Zürich
    
    :param filename: the name of the images (string)
    :param num_images: the number of images
    :param IMG_PATCH_SIZE: the size of the patches
    :return: an array containing the label of all images
    """
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.4d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]

    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)

def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Convert array of labels to an image
    
    Function from the tf_aerial_images.py given on CrowAI
    Credits: Aurelien Lucchi, ETH Zürich
    
    :param imgwidth: the width of the image we want
    :param imgheight: the height of the image we want
    :param w: the width of a patch
    :param h: the height of a patch
    :param labels: the labels we want to convert into a image
    :return: the corresponding image
    """
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels