# Main Modules
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# =========================================================================
# SVM Classifier
# =========================================================================
def svm(feature_arr, label_arr, kernel='linear' , C=1.0, gamma=0.008):
		# Receive the feature array and label array
		X = np.array(feature_arr)
		y = np.array(label_arr)

		# Split the dataset into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# Train the SVM classifier
		svm_classifier = SVC(kernel=kernel, C=C, gamma=gamma)
		svm_classifier.fit(X_train, y_train)

		pickle.dump(svm_classifier, open('classifier.pkl', 'wb'))
		
		return svm_classifier, X_test, y_test
# =========================================================================