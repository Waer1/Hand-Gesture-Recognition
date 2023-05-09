import cv2
from skimage.feature import hog
import numpy as np
from skimage.feature import local_binary_pattern

# =========================================================================
# HOG Feature Extraction
# =========================================================================
def HOG(image, orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (3, 3)):
    # Calculate the HOG features
    hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
    
    # Pad the feature vector with zeros to make sure they all have the same length
    max_size = hog_features.shape[0]
    hog_features = np.pad(hog_features, (0, max_size - hog_features.shape[0]), mode='constant')
    hog_features = np.ravel(hog_features)
    return hog_features

# =========================================================================


# =========================================================================
# LBP Feature Extraction
# =========================================================================
def LBP(image, radius = 1, method = 'uniform'):
    # Define LBP parameters
    n_points = 8 * radius
    
    # Calculate the LBP features
    lbp = local_binary_pattern(image, n_points, radius, method)

    histogram, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Normalize the histogram
    histogram = histogram.astype("float")
    histogram /= (histogram.sum() + 1e-7)

    # The resulting histogram is the feature vector for the input image
    return histogram
# =========================================================================


# =========================================================================
# SIFT Feature Extraction
def SIFT(image, feature_arr):
    # Initialize the SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Extract SIFT features from all the images
    kp, des = sift.detectAndCompute(image, None)
    feature_arr
    return des
# =========================================================================


# =========================================================================
# SURF Feature Extraction
# =========================================================================

def SURF(image):
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(image, None)

    feature_list = []
    for kp, desc in zip(keypoints, descriptors):
        feature = np.concatenate((kp.pt, kp.size, kp.angle, kp.response, kp.octave, desc))
        feature_list.append(feature)
    
    if len(feature_list) == 0:
        return None
    
    feature_arr = np.array(feature_list, dtype=object)
    return feature_arr


# =========================================================================
# normalize features
# =========================================================================

def normalize_features(feature_arrays):
    # Find the maximum size of the feature arrays
    max_size = max(len(arr) for arr in feature_arrays)

    # Pad the feature arrays with zeros
    padded_feature_arrays = []
    for arr in feature_arrays:
        if len(arr) < max_size:
            padded_arr = np.pad(arr, ((0, max_size - len(arr)), (0, 0)), mode='constant')
        else:
            padded_arr = arr
        padded_feature_arrays.append(padded_arr)

    return padded_feature_arrays

# =========================================================================
import os
import cv2
from preprocessing import preprocess
import numpy as np


def HOG_MAIN(images_dir):
    feature_arr = []
    label_arr = []

    for path in images_dir:
        # get all the image names
        images = os.listdir(f"./Dataset/{path}")
        
            
        # iterate over the image names, get the label
        for image in images:
            image_path = f"./Dataset/{path}/{image}"

            try:

                image = cv2.imread(image_path)

                # Preprocessing phase
                image = preprocess(image)

                # Feature extraction phase
                feature = HOG(image)

                # update the data and labels
                feature_arr.append(feature)
                label_arr.append(path)
            except:
                  print(image_path)

    return feature_arr, label_arr


def LBP_MAIN(images_dir):
    feature_arr = []
    label_arr = []

    for path in images_dir:
        # get all the image names
        images = os.listdir(f"./Dataset/{path}")
        
        # iterate over the image names, get the label
        for image in images:
            image_path = f"./Dataset/{path}/{image}"

            try:
                image = cv2.imread(image_path)

                # Preprocessing phase
                image = preprocess(image)

                # Feature extraction phase
                feature = LBP(image)

                # update the data and labels
                feature_arr.append(feature)
                label_arr.append(path)
            except:
                  print(image_path)

    return feature_arr, label_arr


import os
import cv2
import numpy as np

def SIFT_MAIN(images_dir):
    descriptors_list = []
    label_list = []
    max_length = 0

    for path in images_dir:
        # get all the image names
        images = os.listdir(f"./Dataset/{path}")

        # iterate over the image names, get the label
        for image in images:
            image_path = f"./Dataset/{path}/{image}"

            # Read image
            image = cv2.imread(image_path)

            # Preprocessing phase
            image = preprocess(image)

            # SIFT feature extraction
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(image, None)

            if descriptors.shape[0] > max_length:
                max_length = descriptors.shape[0]

            # Append feature and label to respective lists
            descriptors_list.append(descriptors)
            label_list.append(path)

    for i in range(len(descriptors_list)):
        descriptors = descriptors_list[i]
        if descriptors.shape[0] < max_length:
            padding = np.zeros((max_length - descriptors.shape[0], descriptors.shape[1]), dtype=np.float32)
            descriptors = np.vstack((descriptors, padding))
            descriptors_list[i] = descriptors
    
    
    descriptors_list = np.array(descriptors_list)

    nsamples, nx, ny = descriptors_list.shape
    d2_train_dataset = descriptors_list.reshape((nsamples,nx*ny))

    return d2_train_dataset, np.array(label_list)




def ORB_MAIN(images_dir):
    descriptors_list = []
    label_list = []
    max_length = 0

    for path in images_dir:
        # get all the image names
        images = os.listdir(f"./Dataset/{path}")

        # iterate over the image names, get the label
        for image in images:
            image_path = f"./Dataset/{path}/{image}"

            try:
                # Read image
                image = cv2.imread(image_path)

                # Preprocessing phase
                image = preprocess(image)

                # SIFT feature extraction
                orb = cv2.ORB_create()
                keypoints, descriptors = orb.detectAndCompute(image, None)

                if descriptors.shape[0] > max_length:
                    max_length = descriptors.shape[0]

                # Append feature and label to respective lists
                descriptors_list.append(descriptors)
                label_list.append(path)

            except:
                print(image_path)

    for i in range(len(descriptors_list)):
        descriptors = descriptors_list[i]
        if descriptors.shape[0] < max_length:
            padding = np.zeros((max_length - descriptors.shape[0], descriptors.shape[1]), dtype=np.float32)
            descriptors = np.vstack((descriptors, padding))
            descriptors_list[i] = descriptors
    
    
    descriptors_list = np.array(descriptors_list)

    nsamples, nx, ny = descriptors_list.shape
    d2_train_dataset = descriptors_list.reshape((nsamples,nx*ny))

    return d2_train_dataset, np.array(label_list)


def get_feature(FEATURE_METHOD, images_dir):
		if FEATURE_METHOD == 0:
				return HOG_MAIN(images_dir)
		elif FEATURE_METHOD == 1:
				return LBP_MAIN(images_dir)
		elif FEATURE_METHOD == 2:
				return SIFT_MAIN(images_dir)
		elif FEATURE_METHOD == 3:
				return ORB_MAIN(images_dir)
                