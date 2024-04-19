# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle


# 2. Create test set if you like to do the split programmatically or if you have not already split the data at this point
df_test = pd.read_csv('pulsar_stars_test.csv')


# 3. Load your saved model for dissadvantaged communities classification 
#that you saved in dissadvantaged_communities_classification.py via Pikcle
with open('pulsarClassifier.sav', 'rb') as f:
    classifier = pickle.load(f)


# 4. Make predictions on test_set created from step 2
# Separate predictor variables from the target variable
X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Predict target variable using trained model
y_pred = classifier.predict(X_test)


# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.
# Compute and print accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)

# Get and print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)


# Get and print confusion matrix
#cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0, 0]
TN = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]

precision = TP / (TP + FP)
print("Precision:", precision)
