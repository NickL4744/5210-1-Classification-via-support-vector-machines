import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

# Load test dataset
df_test = pd.read_csv('disadvantaged_communities.csv')

# Load trained model
with open('rfClassifier.sav', 'rb') as f:
    classifier = pickle.load(f)

# Separate predictor variables from the target variable
X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Predict target variable using trained model
y_pred = classifier.predict(X_test)

# Compute and print accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)

# Get and print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0, 0]
TN = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]

# Compute Precision and print it
precision = TP / (TP + FP)
print("Precision:", precision)