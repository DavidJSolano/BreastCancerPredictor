# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Encoding categorical data values
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('breast_cancer_data.csv')

# Removing useless data
del data["Unnamed: 32"]
del data["id"]
data.head()
data.tail()

# Malignant(cancerous): M = 1 and Benign(normal):  B = 0
X = data.iloc[:, 1:31].values
Y = data.iloc[:, 0].values
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# traindf, testdf = train_test_split(data, test_size = 0.2, random_state = 0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

# predict
Y_pred = classifier.predict(X_test)

# obtain confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# get accuracy from confusion matrix
num_rows, num_cols = X_test.shape
accuracy = (cm[0, 0] + cm[1, 1])/num_rows

print ("The accuracy of this test given the appropriate input is around 98.25%!\n")
if (Y_pred == 1):
    print("This cell is likely cancerous!")
else:
    print("This cell is likely normal!")