import numpy as np
from skimage.feature import hog

# =========================================================================
# HOG Feature Extraction
# =========================================================================
def hog_features(image, orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (3, 3)):
    # Calculate the HOG features
    hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
    
    # Pad the feature vector with zeros to make sure they all have the same length
    max_size = hog_features.shape[0]
    hog_features = np.pad(hog_features, (0, max_size - hog_features.shape[0]), mode='constant')
    hog_features = np.ravel(hog_features)
    return hog_features
# =========================================================================