import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 1. Load saved classifiers
with open('SvmClassifier.sav', 'rb') as file:
    svm_classifier = pickle.load(file)

with open('RfClassifier.sav', 'rb') as file:
    rf_classifier = pickle.load(file)

# 2. Load dataset and create a test set with 90%, 10% split
data = pd.read_csv('disadvantaged_communities.csv')

# Identify the target variable column
target_column = 'CES 4.0 Percentile Range'  # Replace 'Actual_Target_Column_Name' with your actual target column name

feature_columns = data.columns.drop(target_column)

# Load the test data
test_data = pd.read_csv('disadvantaged_communities.csv')  # Replace with your test dataset filename

# Ensure that the test data contains the same features as the training data
test_data = test_data[feature_columns]

X_test = test_data
y_test = pd.read_csv('disadvantaged_communities.csv')[target_column]  # Replace with your test labels filename

# 3. Predict classes for the test cases
svm_predictions = svm_classifier.predict(X_test)
rf_predictions = rf_classifier.predict(X_test)

# Compute and print evaluation metrics for SVM classifier
print("Evaluation metrics for SVM Classifier:")
print("--------------------------------------------------")

# Compute confusion matrix
cm_svm = confusion_matrix(y_test, svm_predictions)
print("Confusion Matrix:\n", cm_svm)

# Compute accuracy
accuracy_svm = accuracy_score(y_test, svm_predictions)
print("Accuracy Score:", accuracy_svm)

# Compute Precision
precision_svm = precision_score(y_test, svm_predictions, average='macro')
print("Precision:", precision_svm)

# Compute Recall
recall_svm = recall_score(y_test, svm_predictions, average='macro')
print("Recall:", recall_svm)

# Compute Specificity
TN = cm_svm[0, 0]
FP = cm_svm[0, 1]
specificity_svm = TN / (TN + FP)
print("Specificity:", specificity_svm)

# Compute and print evaluation metrics for Random Forest classifier
print("\nEvaluation metrics for Random Forest Classifier:")
print("--------------------------------------------------")

# Compute confusion matrix
cm_rf = confusion_matrix(y_test, rf_predictions)
print("Confusion Matrix:\n", cm_rf)

# Compute accuracy
accuracy_rf = accuracy_score(y_test, rf_predictions)
print("Accuracy Score:", accuracy_rf)

# Compute Precision
precision_rf = precision_score(y_test, rf_predictions, average='macro')
print("Precision:", precision_rf)

# Compute Recall
recall_rf = recall_score(y_test, rf_predictions, average='macro')
print("Recall:", recall_rf)

# Compute Specificity
TN = cm_rf[0, 0]
FP = cm_rf[0, 1]
specificity_rf = TN / (TN + FP)
print("Specificity:", specificity_rf)