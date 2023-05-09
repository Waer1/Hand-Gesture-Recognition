import pickle
import cv2
import numpy as np
import os
from preprocessing import preprocess
from features import get_feature

# =========================================================================
# Get the environment variables
# =========================================================================
# Possible values [0: "HOG", 1: "LBP", 2: "SIFT", 3: "SURF"]
FEATURE_METHOD = os.environ.get('FEATURE_METHOD')

# Possible values ["svm": "SVM", "rb": "RandomForest", "nb": "NaiveBayes", "dt": "DecisionTree"]
MODEL_NAME = os.environ.get('MODEL_NAME')

# Test the model on the test set or on a single image
# Possible values ["test": "Test", "image": "Image"]
TEST_METHOD = os.environ.get('TEST_METHOD')
# =========================================================================

# Load the trained model from pickle file
with open(f'{MODEL_NAME}_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

if TEST_METHOD == "image":
		# Load and preprocess the image
		label = "0"
		image_path = './Dataset/0/1315.jpg'
		image = cv2.imread(image_path)

		# Preprocessing phase
		image = preprocess(image)

		# Compute HOG features for the image
		feature = get_feature(FEATURE_METHOD, image)

		# Make prediction
		predicted_class = model.predict(np.array(feature).reshape(1, -1))

		# Evaluate the prediction
		if predicted_class[0] == label:
				print("Correct prediction")
		else:
				print("Wrong prediction")
else:
		images_number = 0
		error_count = 0

		# get all the image folder paths
		image_paths = os.listdir("./Testset/")

		for path in image_paths:
			# get all the image names
			images = os.listdir(f"./Testset/{path}")
			
			# iterate over the image names, get the label
			for image in images:
				image_path = f"./Testset/{path}/{image}"
				image = cv2.imread(image_path)
				images_number += 1

				# Preprocessing phase
				image = preprocess(image)

				# Feature extraction phase
				feature = get_feature(FEATURE_METHOD, image)

				# Make prediction
				predicted_class = model.predict(feature)

				# Update the error count
				if predicted_class[0] != path:
					error_count += 1

		print("Error:", error_count/images_number*100)
		print("Accuracy:", 100 - error_count/images_number*100)