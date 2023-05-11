# Main Modules
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# SVM
from sklearn.svm import SVC

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# =========================================================================
# SVM Classifier
# =========================================================================
def svm(feature_arr, label_arr, kernel='linear' , C=1.0, gamma=0.008):
		# Receive the feature array and label array
		X = np.array(feature_arr)
		y = np.array(label_arr)

		# Split the dataset into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

		# Train the SVM classifier
		svm_classifier = SVC(kernel=kernel, C=C , gamma=gamma)
		svm_classifier.fit(X_train, y_train)

		# Save the model as a pickle file
		pickle.dump(svm_classifier, open('svm_classifier.pkl', 'wb'))

		# Evaluate the classifier
		y_pred = svm_classifier.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)

		print(f'Accuracy: {accuracy}')
		return accuracy
# =========================================================================


# =========================================================================
# Random Forest Classifier
# =========================================================================
def random_forest(feature_arr, label_arr):
		# Receive the feature array and label array
		X = np.array(feature_arr)
		y = np.array(label_arr)

		# Split the dataset into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# Train the Random Forest classifier
		rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt')
		rf_classifier.fit(X_train, y_train)

		# Save the model as a pickle file
		pickle.dump(rf_classifier, open('rf_classifier.pkl', 'wb'))

		# Evaluate the classifier
		y_pred = rf_classifier.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		print(f'Accuracy: {accuracy}')
# =========================================================================


# =========================================================================
# Naive Bayes Classifier
# =========================================================================
def naive_bayes(feature_arr, label_arr):
		# Receive the feature array and label array
		X = np.array(feature_arr)
		y = np.array(label_arr)

		# Split the dataset into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# Train the Naive Bayes classifier
		nb_classifier = GaussianNB()
		nb_classifier.fit(X_train, y_train)

		# Save the model as a pickle file
		pickle.dump(nb_classifier, open('nb_classifier.pkl', 'wb'))
		
		# Evaluate the classifier
		y_pred = nb_classifier.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		print(f'Accuracy: {accuracy}')
# =========================================================================


# =========================================================================
# Decision Tree Classifier
# =========================================================================
def decision_tree(feature_arr, label_arr):
		# Receive the feature array and label array
		X = np.array(feature_arr)
		y = np.array(label_arr)

		# Split the dataset into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# Train the Decision Tree classifier
		dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
		dt_classifier.fit(X_train, y_train)

		# Save the model as a pickle file
		pickle.dump(dt_classifier, open('dt_classifier.pkl', 'wb'))

		# Evaluate the classifier
		y_pred = dt_classifier.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		print(f'Accuracy: {accuracy}')
# =========================================================================

def get_model(MODEL_METHOD, feature_arr, label_arr):
		if MODEL_METHOD == 0:
				svm(feature_arr, label_arr)
		elif MODEL_METHOD == 1:
				random_forest(feature_arr, label_arr)
		elif MODEL_METHOD == 2:
				naive_bayes(feature_arr, label_arr)
		elif MODEL_METHOD == 3:
				decision_tree(feature_arr, label_arr)