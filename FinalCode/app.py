import pickle
import cv2
import numpy as np
import os
import time
from preprocessing import preprocess
from features import get_feature

# =========================================================================
# Get the environment variables
# =========================================================================
# Possible values [0: "HOG", 1: "LBP", 2: "SIFT", 3: "SURF"]
FEATURE_METHOD = os.environ.get('FEATURE_METHOD')
# =========================================================================

# Load the trained model from pickle file
with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Create the time and results txt files
resultsFile = open("results.txt", "w")
timeFile = open("time.txt", "w")

# get all the image names
images = os.listdir("./data/")

# iterate over the image names, get the label
for image in images:
		image_path = f"./data/{image}"
		image = cv2.imread(image_path)

		# Start the timer
		start_time = time.time()

		# Preprocessing phase
		image = preprocess(image)

		# Feature extraction phase
		feature = get_feature(FEATURE_METHOD, image)

		# Make prediction and output to the file
		predicted_class = model.predict(feature)
		resultsFile.write(f"{predicted_class[0]}\n")

		# Finish the timer and output to the file
		end_time = time.time()
		timeFile.write(f"{round(end_time - start_time, 3)}\n")