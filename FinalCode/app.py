import pickle
import cv2
import argparse
import os
import time
from preprocessing import preprocess
from features import get_feature

# =========================================================================
# Get the environment variables
# =========================================================================
# Create argument parser
parser = argparse.ArgumentParser()
# Add feature argument
parser.add_argument('--feature', type=int, help='feature value')
# Parse arguments
args = parser.parse_args()

# Possible values [0: "HOG", 1: "LBP", 2: "SIFT", 3: "SURF"]
# Read feature
FEATURE_METHOD = args.feature
# =========================================================================

# Load the trained model from pickle file
with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Create the time and results txt files
resultsFile = open("results.txt", "w")
timeFile = open("time.txt", "w")

# get all the image names
dataset = "./data/"
images = os.listdir(dataset)

# iterate over the image names, get the label
for image in images:
		image_path = dataset + image
		image = cv2.imread(image_path)

		# Start the timer
		start_time = time.time()

		try:
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
		except:
				resultsFile.write(f"Error in image {image}\n")
				timeFile.write(f"Error in image {image}\n")