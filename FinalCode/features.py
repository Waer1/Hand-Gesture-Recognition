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
# =========================================================================
def SIFT(image):
    # assume this image to be gray scale
    # Create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Initialize empty feature list
    features = []

    # Loop through each keypoint and descriptor pair
    for i in range(len(keypoints)):
        # Get current keypoint and descriptor
        kp = keypoints[i]
        desc = descriptors[i]

        # Convert keypoint to feature vector
        feature = np.concatenate((kp.pt, kp.size, kp.angle, kp.response, kp.octave, desc))
        
        # Append feature to feature list
        features.append(feature)
        
    # Convert features list to NumPy array
    features = np.array(features)
    return features
# =========================================================================


# =========================================================================
# SURF Feature Extraction
# =========================================================================
def SURF(image):
    # assume this image to be gray scale
    # Create a SURF object
    surf = cv2.xfeatures2d.SURF_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = surf.detectAndCompute(image, None)

    # Initialize empty feature list
    features = []

    # Loop through each keypoint and descriptor pair
    for i in range(len(keypoints)):
        # Get current keypoint and descriptor
        kp = keypoints[i]
        desc = descriptors[i]

        # Convert keypoint to feature vector
        feature = np.concatenate((kp.pt, kp.size, kp.angle, kp.response, kp.octave, desc))
        
        # Append feature to feature list
        features.append(feature)
        
    # Convert features list to NumPy array
    features = np.array(features)
    return features
# =========================================================================

def get_feature(FEATURE_METHOD, image):
		if FEATURE_METHOD == 0:
				return HOG(image)
		elif FEATURE_METHOD == 1:
				return LBP(image)
		elif FEATURE_METHOD == 2:
				return SIFT(image)
		elif FEATURE_METHOD == 3:
				return SURF(image)