#step 1
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

#step 2
# Load dataset and show basic statistics
df = pd.read_csv('DACs_Dictionary.csv')

# 1. Show dataset size (dimensions)
print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Show what column names exist for the 9 attributes in the dataset
print(f"Column names: {df.columns.tolist()}")

# 3. Show the distribution of target_class column
print(df['target_class'].value_counts())

# 4. Show the percentage distribution of target_class column
print(df['target_class'].value_counts(normalize=True) * 100)


#step 5
# Separate predictor variables from the target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Create train and test splits for model development. Use the 80% and 20% split ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#step 6
# Standardize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM with the following parameters:
# 1. RBF kernel
# 2. C=10.0 (Higher value of C means fewer outliers)
# 3. gamma 0.3
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=10.0, gamma=0.3)
svm.fit(X_train, y_train)

# Save SVM model
filename = 'SVMClassifier.sav'
pickle.dump(svm, open(filename, 'wb'))

# Test the above developed SVM on unseen pulsar dataset samples
y_pred = svm.predict(X_test)

# Print accuracy score
accuracy = svm.score(X_test, y_test)
print(f"Accuracy score: {accuracy:.3f}")

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = scaler.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, svm.predict(scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support vector machine (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = scaler.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, svm.predict(scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support vector machine (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Get and print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Compute and print precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.3f}")

# Compute and print recall or sensitivity
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print(f"Recall or Sensitivity: {recall:.3f}")

# Compute and print specificity
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.3f}")


