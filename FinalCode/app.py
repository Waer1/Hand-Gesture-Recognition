import pickle
import cv2
import os
import time
import numpy as np
from preprocessing import preprocess
from features import hog_features

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
				feature = np.array(hog_features(image)).reshape(1, -1)

				# Make prediction and output to the file
				predicted_class = model.predict(feature)
				
				resultsFile.write(f"{predicted_class[0]}\n")

				# Finish the timer and output to the file
				end_time = time.time()
				timeFile.write(f"{round(end_time - start_time, 3)}\n")
		except:
				resultsFile.write(f"Error in image {image_path}\n")
				timeFile.write(f"Error in image {image_path}\n")