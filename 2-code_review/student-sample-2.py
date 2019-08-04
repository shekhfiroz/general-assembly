#!/usr/bin/env python

# please follow below steps

# Import libraries
# Read data
# Checking for missing values
# Checking for categorical data
# Data splitting
# build the model
# train the model
# test the model and check the accuracy

#import necssary libaray
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import LinearRegression
from sklearn.cross_validation import cross_val_score

# Load data
#d = pd.read_csv('../data/train.csv')

# d is ambigious so  take  relevent variable name
data = pd.read_csv('../data/train.csv')

# analyse the top 5 records of dataframe
data.head()

#analyse the last 5 records of dataframe
data.tail()

#check the count of rows and columns of the dataset
data.shape

#check the missing values in dataset
data.isnull().any().sum()

#set categorical data to binary data 0 and 1
#Change all 'M' to 1 and
#Change all 'B' to 0 in the diagnosis col
#Encoding categorical data values (Transforming categorical data/Strings to integers)
labelencoder_Y = LabelEncoder()
data.iloc[:, 1] = labelencoder_Y.fit_transform(data.iloc[:, 1].values)
print(labelencoder_Y.fit_transform(data.iloc[:, 1].values))

#find median
data.median

#find mode
data.mode

#Get a count of the number of Malignant (M) (harmful) or Benign (B) cells (not harmful)
#all 'M' to 1 and
#all 'B' to 0 in the diagnosis col
# we can see Malignant(M) Which is harmful is less than the Benign cells in our data set
data['diagnosis'].value_counts()

#Get a sample of correlated column info
data.iloc[:, 1:10].corr()

#we can visualize correlation
# To see the numbers within the cell ==>sns.heatmap(df.corr(), annot=True)
#This is used to change the size of the figure/ heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, fmt='.0%',)

#Split the data into independent 'X1' and dependent 'Y1' variables
#started from index  2 to 31, essentially removing the id column & diagnosis
X1 = data.iloc[:, 2:31].values

#Get the target variable 'diagnosis' located at index=1
Y1 = data.iloc[:, 1].values

# Split the dataset into 75% Training set and 25% Testing set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# Scale the data to bring all features to the same level of magnitude
#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# define the model  and write the code with proper dependent variable and independent variable
#Using Logistic Regression Algorithm to the Training Set


def models(X_train, Y_train):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    return log


#trainning our models by trainning dataset
model = models(X_train, Y_train)

# Evaluate model
scores = cross_val_score(model, x2, x1, cv=1, scoring='mean_absolute_error')
print(scores.mean())

#heat map of confusion matrix
sns.heatmap(cm, annot=True)

#confusion matrix and accuracy for all of the models on the test data
#Classification accuracy is the ratio of correct predictions to total predictions made.
#TN =true negative,
#TP =true positive,
#FN=false negative,
#FP=false positive
for i in range(len(model)):
  cm = confusion_matrix(Y_test, model[i].predict(X_test))
  TN = cm[0][0]
  TP = cm[1][1]
  FN = cm[1][0]
  FP = cm[0][1]

 print(cm)
 
 # you will get the accuracy of the model
