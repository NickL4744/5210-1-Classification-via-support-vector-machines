#Step 1:
#Import libraries
#pip install pandas, matplotlib, and scikit-learn

import  numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#Step 2:
# Load dataset and show basic statistics
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#Split dataset in train and test splits
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Data preprocessing
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Model Development (Support Vector Classification)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

#accuracy test
print(classifier.predict(sc.transform([[30, 87000]])))

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
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
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
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


# Step 3:
#Clean the dataset - you can eitherhandle the missing values in the dataset
# with the mean of the columns attributes or remove rows the have missing values.


# Step 4: 
#Encode the Categorical Variables - Using OrdinalEncoder from the category_encoders library to encode categorical variables as ordinal integers


# Step 5: 
# Separate predictor variables from the target variable (attributes (X) and target variable (y) as we did in the class)
# Create train and test splits for model development. Use the 90% and 20% split ratio
# Use stratifying (stratify=y) to ensure class balance in train/test splits
# Name them as X_train, X_test, y_train, and y_test
# Name them as X_train, X_test, y_train, and y_test
X_train = [] # Remove this line after implementing train test split
X_test = [] # Remove this line after implementing train test split


# Do not do steps 6 - 8 for the Ramdom Forest Model
# Step 6:
# Standardize the features (Import StandardScaler here)



# Step 7:
# Below is the code to convert X_train and X_test into data frames for the next steps
cols = X_train.columns
X_train = pd.DataFrame(X_train, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd



# Step 8 - Build and train the SVM classifier
# Train SVM with the following parameters. (use the parameters with the highest accuracy for the model)
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3 (Linear)



# Test the above developed SVC on unseen pulsar dataset samples

# compute and print accuracy score



# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment




# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix

cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# Compute Precision and use the following line to print it
precision = 0 # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
recall = 0 # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = 0 # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))






# Step 9: Build and train the Random Forest classifier
# Train Random Forest  with the following parameters.
# (n_estimators=10, random_state=0)

# Test the above developed Random Forest model on unseen DACs dataset samples

# compute and print accuracy score

# Save your Random Forest model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# Compute Classification Accuracy and use the following line to print it
classification_accuracy = 0
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# Compute Precision and use the following line to print it
precision = 0 # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
recall = 0 # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = 0 # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))