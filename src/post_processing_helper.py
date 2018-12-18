"""
This file contains several functions to transform and modify images

Benjamin Délèze, Cedric Maire, Antonio Morais
"""
import numpy as np

def suppress_single_roads(labels, size_width, size_height):
    """
    Given an image, turn the single road patches circled by background into background
    
    :param labels: the labels of the image
    :param size_width: the width of the image
    :param size_height: the height of the image
    :return new_labels: the new labels for the image
    """
    new_labels = np.copy(labels)
    for i in range(1, size_width - 1):
        for j in range(1, size_height - 1):
            if labels[get_index(i, j, size_width)][0] <= 0.5: #road
                if labels[get_index(i + 1, j, size_width)][0] > 0.5 and labels[get_index(i - 1, j, size_width)][0] > 0.5 and labels[get_index(i, j + 1, size_width)][0] > 0.5 and labels[get_index(i, j - 1, size_width)][0] > 0.5: #surrounded by bgrd
                        new_labels[get_index(i, j , size_width)][0] = 0.8 # the new label will then be bgrd
    return new_labels

def get_index(i, j, size_width):
    """
    Given the position i and j of a patch return its corresponding index in an array of size size_width
    
    :param i: index i of the image
    :param j: index j of the image
    :param size_width: width of the image
    :return: the corresponding index in the array
    """
    
    return j * size_width + i

def add_road_when_line_of_roads(labels, size_width, size_height, number_of_road_needed):
    """
    Turn background's patches into road's ones when the are in the middle of a line of road patches
    
    :param labels: the labels of the image
    :param size_width: the width of the image
    :param size_height: the height of the image
    :param number_of_road_needed: number of road's patches that we need to have on each side of a patch such that we convert it
    :return new_labels: the new labels of the image
    """
    new_labels = np.copy(labels)
    first = False
    #horizontal lines
    for i in range(0, size_width):
        for j in range(number_of_road_needed, size_height - number_of_road_needed):
            if labels[get_index(i, j, size_width)][0] > 0.5: #bgrd
                surroundedByRoad = True
                for idx in range(1, number_of_road_needed + 1):
                    if labels[get_index(i , j - idx, size_width)][0] > 0.5:
                        surroundedByRoad = False
                        break
                    if labels[get_index(i , j + idx, size_width)][0] > 0.5:
                        surroundedByRoad = False
                        break
                if surroundedByRoad: 
                    new_labels[get_index(i, j, size_width)][0] = 0.2
    #vertical lines                
    for j in range(0, size_height):
        for i in range(number_of_road_needed, size_width - number_of_road_needed):
            if labels[get_index(i, j, size_width)][0] > 0.5: #bgrd
                surroundedByRoad = True
                for idx in range(1, number_of_road_needed + 1):
                    if labels[get_index(i - idx, j , size_width)][0] > 0.5:
                        surroundedByRoad = False
                        break
                    if labels[get_index(i+ idx, j , size_width)][0] > 0.5:
                        surroundedByRoad = False
                        break
                if surroundedByRoad: 
                    new_labels[get_index(i, j, size_width)][0] = 0.2
    return new_labels

def suppress_group_roads_surrounded(labels, size_width, size_height, size_square):
    """
    Look at each square of size_square. If the square contains some road's patches and the square is surrounded only by background patches, transfoms the road's patches into background's ones. 
    
    :param labels: labels of the image
    :param size_width: width of the image
    :param size_height: height of the image
    :param size_square: size of the inner square considered
    :return new_labeles: the new labels of the image
    """
    new_labels = np.copy(labels)
    for i in range(1, size_width - size_square):
        for j in range(1, size_height - size_square):
            #We verify that one of the patches from the square is road
            one_patch_is_road = labels[get_index(i, j, size_width)][0] <= 0.5 
            for k1 in range(1, size_square):
                for k2 in range(1, size_square):
                    one_patch_is_road = one_patch_is_road or labels[get_index(i + k1, j + k2, size_width)][0] <= 0.5 
            if one_patch_is_road:
                #We verify that the outside part is indeed background
                outside_part = True
                for k in range(-1, size_square + 1):
                    outside_part = outside_part and labels[get_index(i - 1, j + k, size_width)][0] > 0.5 
                    outside_part = outside_part and labels[get_index(i + size_square, j + k, size_width)][0] > 0.5 
                    outside_part = outside_part and labels[get_index(i + k, j - 1, size_width)][0] > 0.5 
                    outside_part = outside_part and labels[get_index(i + k, j + size_square, size_width)][0] > 0.5 
                if outside_part:
                    for k1 in range(0, size_square):
                        for k2 in range(0, size_square):
                            new_labels[get_index(i + k1, j + k2 , size_width)][0] = 0.8
    return new_labels