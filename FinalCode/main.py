# Imports
import cv2
from preprocessing import preprocess
from features import hog_features
from models import svm
from performance import performanceAnalysis
import os


# get all the image folder paths
dataset = "../../Dataset/"
image_paths = os.listdir(dataset)

# Hog parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# SVM parameters
kernal = 'linear'
C = 0.1
gamma = 0.001

feature_arr = []
label_arr = []

for path in image_paths:
		# get all the image names
		images = os.listdir(dataset + path)

		# iterate over the image names, get the label
		for image in images:
				image_path = dataset + f"{path}/{image}"
				try:
						image = cv2.imread(image_path)

						# Preprocessing phase
						image = preprocess(image)

						# Feature extraction phase
						feature = hog_features(image, orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block)

						# update the data and labels
						feature_arr.append(feature)
						label_arr.append(path)
				except:
						print(image_path)


# Train the model
model, X_test, y_test = svm(feature_arr, label_arr, kernel = kernal, C = C, gamma = gamma)

# Test the model
performanceAnalysis(model, X_test, y_test)