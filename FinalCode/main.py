# Imports
import cv2
from features import get_feature
from models import get_model
import argparse
import os
import numpy as np

# Create argument parser
parser = argparse.ArgumentParser()
# Add feature argument
parser.add_argument('--feature', type=int, help='feature value')
# Add model argument
parser.add_argument('--model', type=int, help='model value')
# Parse arguments
args = parser.parse_args()

# Read feature and model values
FEATURE_METHOD = args.feature
MODEL_METHOD = args.model

# get all the image folder paths
dataset = "./Dataset/"
image_paths = os.listdir(dataset)

feature_arr , label_arr = get_feature(FEATURE_METHOD , image_paths)

get_model(MODEL_METHOD, feature_arr, label_arr)