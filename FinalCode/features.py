import cv2
from skimage.feature import hog
import numpy as np
from skimage import data, exposure
from skimage.feature import local_binary_pattern

# =========================================================================
# HOG Feature Extraction
# =========================================================================
def HOG(image, orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (3, 3), visualize = True):
    # Define HOG parameters

    # Calculate the HOG features
    hog_features, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=visualize)

    return hog_features
    # # Rescale the image for better visualization
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # # Show the original image and HOG features
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
# =========================================================================


# =========================================================================
# LBP Feature Extraction
# =========================================================================
def LBP(image, radius = 1, method = 'uniform'):
    # Define LBP parameters
    n_points = 8 * radius
    
    # Calculate the LBP features
    lbp = local_binary_pattern(image, n_points, radius, method)

    # # Calculate the histogram of LBP features
    # n_bins = int(lbp.max() + 1)
    # hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    # # Normalize the histogram
    # hist /= hist.sum()
    return lbp
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

    # Draw keypoints on the image
    # img = cv2.drawKeypoints(image, keypoints, img)
    return keypoints, descriptors
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

    # Draw keypoints on the image
    # img = cv2.drawKeypoints(image, keypoints, img)
    return keypoints, descriptors
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