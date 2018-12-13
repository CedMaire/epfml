import numpy as np
import cv2
import os
import re
import matplotlib.image as mpimg
from matplotlib import cm
from PIL import Image

ROOT_DIR = "data/training/"
IMAGE_DIR = ROOT_DIR + "images/"
IMAGE_TEST_DIR = "data/test/"

GROUNDTRUTH_DIR = ROOT_DIR + "groundtruth/"

STRING_SAT_IMAGE_ = "satImage_"
STRING_PNG_EXT = ".png"
STRING_TEST_ = "test_"

ANGLE_90 = 90
ANGLE_180 = 180
ANGLE_270 = 270

SCALE = 1.0

FLIP_CODE_X = 0
FLIP_CODE_Y = 1

def natural_key(string_):
    # https://stackoverflow.com/a/3033342
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def save_images(images, images_groundtruth):
    last_number = int(sorted(os.listdir(IMAGE_DIR), key=natural_key)[-1].lstrip(STRING_SAT_IMAGE_).rstrip(STRING_PNG_EXT))

    for index, image in enumerate(images):
        Image.fromarray(np.uint8(image * 255)).convert("RGB").save(IMAGE_DIR + STRING_SAT_IMAGE_ + str(last_number + index + 1) + STRING_PNG_EXT)

    for index, image_groundtruth in enumerate(images_groundtruth):
        Image.fromarray(np.uint8(image_groundtruth * 255)).convert("L").save(GROUNDTRUTH_DIR + STRING_SAT_IMAGE_ + str(last_number + index + 1) + STRING_PNG_EXT)

def load_images():
    files = sorted(os.listdir(IMAGE_DIR), key=natural_key)

    return [mpimg.imread(IMAGE_DIR + files[i]) for i in range(len(files))], [mpimg.imread(GROUNDTRUTH_DIR + files[i]) for i in range(len(files))]

def save_images_test(images_test):
    last_number = int(sorted(os.listdir(IMAGE_TEST_DIR), key=natural_key)[-1].lstrip(STRING_TEST_))

    for index, image_test in enumerate(images_test):
        if not os.path.isdir(IMAGE_TEST_DIR + str(last_number + index + 1)):
            os.mkdir(IMAGE_TEST_DIR + STRING_TEST_ + str(last_number + index + 1))

        Image.fromarray(np.uint8(image_test * 255)).convert("RGB").save(IMAGE_TEST_DIR + STRING_TEST_ + str(last_number + index + 1) + "/" + STRING_TEST_ + str(last_number + index + 1) + STRING_PNG_EXT)

def load_images_test():
    folders = sorted(os.listdir(IMAGE_TEST_DIR), key=natural_key)

    return [mpimg.imread(IMAGE_TEST_DIR + folders[i] + "/" + folders[i] + STRING_PNG_EXT) for i in range(len(folders))]

def gen_rotated_images(images, images_groundtruth, images_test, save=False):
    images_rotated_90 = []
    images_groundtruth_rotated_90 = []
    images_rotated_180 = []
    images_groundtruth_rotated_180 = []
    images_rotated_270 = []
    images_groundtruth_rotated_270 = []

    images_test_rotated_90 = []
    images_test_rotated_180 = []
    images_test_rotated_270 = []

    for image, image_groundtruth in zip(images, images_groundtruth):
        height, width = image.shape[:2]
        center = (height / 2, width / 2)

        MATRIX_ROT_90 = cv2.getRotationMatrix2D(center, ANGLE_90, SCALE)
        MATRIX_ROT_180 = cv2.getRotationMatrix2D(center, ANGLE_180, SCALE)
        MATRIX_ROT_270 = cv2.getRotationMatrix2D(center, ANGLE_270, SCALE)

        images_rotated_90.append(cv2.warpAffine(image, MATRIX_ROT_90, (height, width)))
        images_rotated_180.append(cv2.warpAffine(image, MATRIX_ROT_180, (height, width)))
        images_rotated_270.append(cv2.warpAffine(image, MATRIX_ROT_270, (height, width)))

        images_groundtruth_rotated_90.append(cv2.warpAffine(image_groundtruth, MATRIX_ROT_90, (height, width)))
        images_groundtruth_rotated_180.append(cv2.warpAffine(image_groundtruth, MATRIX_ROT_180, (height, width)))
        images_groundtruth_rotated_270.append(cv2.warpAffine(image_groundtruth, MATRIX_ROT_270, (height, width)))

    for image_test in images_test:
        height, width = image_test.shape[:2]
        center = (height / 2, width / 2)

        MATRIX_ROT_90 = cv2.getRotationMatrix2D(center, ANGLE_90, SCALE)
        MATRIX_ROT_180 = cv2.getRotationMatrix2D(center, ANGLE_180, SCALE)
        MATRIX_ROT_270 = cv2.getRotationMatrix2D(center, ANGLE_270, SCALE)

        images_test_rotated_90.append(cv2.warpAffine(image_test, MATRIX_ROT_90, (height, width)))
        images_test_rotated_180.append(cv2.warpAffine(image_test, MATRIX_ROT_180, (height, width)))
        images_test_rotated_270.append(cv2.warpAffine(image_test, MATRIX_ROT_270, (height, width)))

    if save:
        save_images(images_rotated_90, images_groundtruth_rotated_90)
        save_images(images_rotated_180, images_groundtruth_rotated_180)
        save_images(images_rotated_270, images_groundtruth_rotated_270)

        save_images_test(images_test_rotated_90)
        save_images_test(images_test_rotated_180)
        save_images_test(images_test_rotated_270)
    else:
        cv2.imshow("Original", images[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rotation 90", images_rotated_90[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rotation 180", images_rotated_180[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rotation 270", images_rotated_270[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Original", images_groundtruth[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rotation 90", images_groundtruth_rotated_90[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rotation 180", images_groundtruth_rotated_180[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rotation 270", images_groundtruth_rotated_270[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Original", images_test[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rotation 90", images_test_rotated_90[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rotation 180", images_test_rotated_180[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rotation 270", images_test_rotated_270[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def flip_images(images, flip_code):
    images_flipped = []

    for image in images:
        images_flipped.append(cv2.flip(src=image, flipCode=flip_code))

    return images_flipped

if __name__ == "__main__":
    images, images_groundtruth = load_images()
    images_test = load_images_test()
    gen_rotated_images(images, images_groundtruth, images_test, save=False)

    images_flipped_x, images_groundtruth_flipped_x = flip_images(images, FLIP_CODE_X), flip_images(images_groundtruth, FLIP_CODE_X)
    images_test_flipped_x = flip_images(images_test, FLIP_CODE_X)
    gen_rotated_images(images_flipped_x, images_groundtruth_flipped_x, images_test_flipped_x, save=False)
