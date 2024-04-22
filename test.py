import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import pickle
import numpy as np

# Step 2:
# Load dataset and show basic statistics
dataset = pd.read_csv('disadvantaged_communities.csv')

print("Dataset size:", dataset.shape)
print("Column names:", dataset.columns)
print("Target class distribution:\n", dataset['CES 4.0 Percentile Range'].value_counts())
print("Target class percentage distribution:\n", dataset['CES 4.0 Percentile Range'].value_counts(normalize=True) * 100)

# Step 3:
dataset.dropna(inplace=True)

# Step 4:
encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(dataset)

# Step 5:
X = data_encoded[:, :-1]
y = data_encoded[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 6:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7:
cols = dataset.columns[:-1]
X_train = pd.DataFrame(X_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)

# Step 8:
svm_classifier = SVC(kernel='rbf', C=10.0, gamma=0.3)
svm_classifier.fit(X_train, y_train)

svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

with open('SVMClassifier.sav', 'wb') as file:
    pickle.dump(svm_classifier, file)

cm_svm = confusion_matrix(y_test, svm_predictions)
TP = cm_svm[0, 0]
TN = cm_svm[1, 1]
FP = cm_svm[0, 1]
FN = cm_svm[1, 0]

precision_svm = precision_score(y_test, svm_predictions, average='weighted')
recall_svm = recall_score(y_test, svm_predictions, average='weighted')
specificity_svm = TN / (TN + FP)

print('SVM Precision:', precision_svm)
print('SVM Recall:', recall_svm)
print('SVM Specificity:', specificity_svm)

# Step 9:
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
rf_classifier.fit(X_train, y_train)

with open('RFClassifier.sav', 'wb') as file:
    pickle.dump(rf_classifier, file)

rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)

cm_rf = confusion_matrix(y_test, rf_predictions)
TP = cm_rf[0, 0]
TN = cm_rf[1, 1]
FP = cm_rf[0, 1]
FN = cm_rf[1, 0]

# Compute Precision, Recall, and Specificity for Random Forest with average='macro'
precision_rf = precision_score(y_test, rf_predictions, average='macro')
recall_rf = recall_score(y_test, rf_predictions, average='macro')

# For specificity, we need to compute it manually for each class and then take the average
# Compute specificity for each class
specificity_rf = []
for i in range(len(np.unique(y_test))):
    TP = np.sum((y_test == i) & (rf_predictions == i))
    TN = np.sum((y_test != i) & (rf_predictions != i))
    FP = np.sum((y_test != i) & (rf_predictions == i))
    FN = np.sum((y_test == i) & (rf_predictions != i))
    
    specificity_i = TN / (TN + FP) if (TN + FP) != 0 else 0
    specificity_rf.append(specificity_i)

# Compute average specificity
specificity_rf = np.mean(specificity_rf)

print('Random Forest Precision:', precision_rf)
print('Random Forest Recall:', recall_rf)
print('Random Forest Specificity:', specificity_rf)