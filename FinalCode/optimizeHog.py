
# Imports
from features import get_feature
from models import get_model
from models import svm
from features import HOG_MAIN
import itertools
from joblib import Parallel, delayed
import os

# get all the image folder paths
dataset = "./Dataset/"
image_paths = os.listdir(dataset)

# get all the image folder paths
dataset = "./Dataset/"
image_paths = os.listdir(dataset)

from joblib import Parallel, delayed

# get all the image folder paths
dataset = "./Dataset/"
image_paths = os.listdir(dataset)

def run_single(orientation, pixels_per_cell, cells_per_block, kernel, C, gamma):
    feature_arr, label_arr = HOG_MAIN(image_paths, dataset, orientation, pixels_per_cell, cells_per_block)
    accuracy = svm(feature_arr, label_arr, kernel, C, gamma)
    print(f"Orientation: {orientation}, Pixels per cell: {pixels_per_cell}, Cells per block: {cells_per_block}, Kernel: {kernel}, C: {C}, Gamma: {gamma}, Accuracy: {accuracy}")
    return accuracy


def run_HOG_SVM(orientations_list, pixels_per_cell_list, cells_per_block_list, kernel_list, C_list, gamma_list):
    best_accuracy = 0
    best_params = {}
    accuracy = 0

    
    best_params['gamma'] = gamma_list[0]
    best_params['C'] = C_list[0]
    best_params['kernel'] = kernel_list[0]
    best_params['cells_per_block'] = cells_per_block_list[0]
    best_params['pixels_per_cell'] = pixels_per_cell_list[0]
    best_params['orientations'] = orientations_list[0]


    for gamma in gamma_list:
        accuracy = run_single(orientations_list[0] , pixels_per_cell_list[0], cells_per_block_list[0], kernel_list[0], C_list[0], gamma)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params['gamma'] = gamma
    
    print("the best gamma parameters were: ", best_params['gamma'])


    for c_val in C_list:
        accuracy = run_single(orientations_list[0] , pixels_per_cell_list[0], cells_per_block_list[0], kernel_list[0], c_val, best_params['gamma'])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params['C'] = gamma

    print("the best C parameters were: ", best_params['C'])

    for kernal_val in kernel_list:
        accuracy = run_single(orientations_list[0] , pixels_per_cell_list[0], cells_per_block_list[0], kernal_val, best_params['C'], best_params['gamma'])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params['kernel'] = kernal_val

    print("the best kernel parameters were: ", best_params['kernel'])

    for cell_per_block_val in cells_per_block_list:
        accuracy = run_single(orientations_list[0] , pixels_per_cell_list[0], cell_per_block_val, best_params['kernel'], best_params['C'], best_params['gamma'])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params['cells_per_block'] = cell_per_block_val

    print("the best cells_per_block parameters were: ", best_params['cells_per_block'])

    for pixels_per_cell_val in pixels_per_cell_list:
        accuracy = run_single(orientations_list[0] , pixels_per_cell_val, best_params['cells_per_block'] ,  best_params['kernel'], best_params['C'], best_params['gamma'])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params['pixels_per_cell'] = pixels_per_cell_val

    print("the best pixels_per_cell parameters were: ", best_params['pixels_per_cell'])

    for orientations_val in orientations_list:
        accuracy = run_single(orientations_val , best_params['pixels_per_cell'], best_params['cells_per_block'] ,  best_params['kernel'], best_params['C'], best_params['gamma'])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params['orientations'] = orientations_val

    print("the best orientations parameters were: ", best_params['orientations'])

    return best_accuracy, best_params
                                

orientations_list = [9, 5, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
pixels_per_cell_list = [(8, 8), (6, 6), (4, 4), (8, 8), (10, 10), (12, 12)]
cells_per_block_list = [(1, 1), (2, 2), (3, 3), (4, 4)]
kernel = ["linear", "poly", "rbf", "sigmoid"]
C = [0.5, 1.5, 0.0001, 0.01, 0.1, 1, 10, 100, 1000]
gamma = ["scale", "auto", 0.1, 1, 0.0001, 0.001, 0.01, 10, 100]

best_accuracy, best_params = run_HOG_SVM(orientations_list, pixels_per_cell_list , cells_per_block_list , kernel , C , gamma)

print(f"Best accuracy: {best_accuracy}")
print(f"Best params: {best_params}")