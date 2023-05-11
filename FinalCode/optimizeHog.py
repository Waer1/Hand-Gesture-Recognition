
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

def run_HOG_SVM(orientations_list, pixels_per_cell_list, cells_per_block_list,kernel, C, gamma ):

    best_accuracy = 0
    best_params = {}

    def run_single(orientations, pixels_per_cell, cells_per_block, c_value, kernel_value, gamma_value):
        feature_arr , label_arr = HOG_MAIN(image_paths, dataset, orientations, pixels_per_cell, cells_per_block)
        accuracy = svm(feature_arr, label_arr, kernel_value, c_value, gamma_value)
        print(f"orientations: {orientations}, pixels_per_cell: {pixels_per_cell}, cells_per_block: {cells_per_block}, kernel: {kernel_value}, C: {c_value} , gamma: {gamma_value} , kernel: {kernel_value} : accuracy: {accuracy} ") 
        return (accuracy, {'orientations': orientations, 'pixels_per_cell': pixels_per_cell,
                           'cells_per_block': cells_per_block, 'C': C, 'kernel': kernel, 'gamma': gamma})

    results = Parallel(n_jobs=8)(delayed(run_single)(orientations, pixels_per_cell, cells_per_block, c_value, kernel_value, gamma_value)
                                       for orientations, pixels_per_cell, cells_per_block, c_value ,kernel_value , gamma_value  in itertools.product(orientations_list, pixels_per_cell_list,
                                                                              cells_per_block_list, C, kernel, gamma))

    for accuracy, params in results:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    return best_accuracy, best_params

orientations_list = [9, 5, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
pixels_per_cell_list = [(8, 8), (6, 6), (4, 4), (8, 8), (10, 10), (12, 12), (4, 4), (6, 6), (8, 8), (10, 10)]
cells_per_block_list = [(1, 1), (2, 2), (3, 3), (4, 4), (2, 2), (3, 3), (4, 4)]
kernel = ["linear", "poly", "rbf", "sigmoid", "rbf", "poly", "sigmoid"]
C = [0.01, 0.5, 1.5, 10, 1000, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
gamma = ["scale", "auto", 0.1, 1, "scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 0.001, 0.01, 0.1, 1, 10, 100]

best_accuracy, best_params = run_HOG_SVM(orientations_list, pixels_per_cell_list , cells_per_block_list , kernel , C , gamma)
print(f"Best accuracy: {best_accuracy}")
print(f"Best params: {best_params}")