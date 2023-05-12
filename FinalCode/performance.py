from sklearn.metrics import accuracy_score

def performanceAnalysis(model, X_test, y_test):
		y_pred = model.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		print(f'Accuracy: {accuracy}')