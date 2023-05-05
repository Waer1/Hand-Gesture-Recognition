# Imports
import cv2
import os
from preprocessing import preprocess
from features import get_feature
from models import get_model


# =========================================================================
# Get the environment variables
# =========================================================================
# Possible values [0: "HOG", 1: "LBP", 2: "SIFT", 3: "SURF"]
# FEATURE_METHOD = os.environ.get('FEATURE_METHOD')
FEATURE_METHOD = 2

# Possible values [0: "SVM", 1: "RandomForest", 2: "NaiveBayes", 3: "DecisionTree"]
# MODEL_METHOD = os.environ.get('MODEL_METHOD')
MODEL_METHOD = 0
# =========================================================================


# print(FEATURE_METHOD , MODEL_METHOD)

feature_arr = []
label_arr = []

# get all the image folder paths
image_paths = os.listdir("./Dataset/")

for path in image_paths:
	# get all the image names
	images = os.listdir(f"./Dataset/{path}")
	
	# iterate over the image names, get the label
	for image in images:
		print(f"Extracting features from {image}...")
		image_path = f"./Dataset/{path}/{image}"
		image = cv2.imread(image_path)

		# Preprocessing phase
		image = preprocess(image)

		# Feature extraction phase
		feature = get_feature(FEATURE_METHOD, image)

		# update the data and labels
		feature_arr.append(feature)
		label_arr.append(path)

# Training phase
get_model(MODEL_METHOD, feature_arr, label_arr)