import pickle
import cv2
import numpy as np
from preprocessing import preprocess
from features import get_feature

# =========================================================================
# Get the environment variables
# =========================================================================
# Possible values [0: "HOG", 1: "LBP", 2: "SIFT", 3: "SURF"]
# FEATURE_METHOD = os.environ.get('FEATURE_METHOD')
FEATURE_METHOD = 0
# =========================================================================

# Load the trained model from pickle file
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and preprocess the image
image_path = 'photo.jpg'
image = cv2.imread(image_path)
image = preprocess(image)

# Compute HOG features for the image
feature = get_feature(0, image)

# Make prediction
predicted_class = model.predict(feature)

# Print the predicted class
print("Predicted class:", predicted_class)
