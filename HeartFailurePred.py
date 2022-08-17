# -*- coding: utf-8 -*-
"""
Created on Mon May  3 07:02:57 2021

@author: Nathaniel Carballo
"""

#importing libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#opening data set for predictions. printing length of data and printing 
#full view of columns and rows.
dataset = pd.read_csv('heart.csv')
print (len(dataset))
print (dataset.head())

#replaces zeroes within the columns listed
zero_not_accepted = ['trtbps', 'chol','thalachh', 'thall']

#replaces zeroes with NaN then grabs mean while skipping NaN.
#replaces NaN with mean
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0,np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)
    
#splits the dataset into train and test groups
x = dataset.iloc[:,0:13]
y = dataset.iloc[:, 13]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.3)

#scales data for less noise and keep data standardized
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

#defines the prediction model used:KNN and fits data within its parameters
classifier = KNeighborsClassifier(n_neighbors=9, p=2, metric='euclidean')
classifier.fit(x_train, y_train)
KNeighborsClassifier(algorithm='auto',leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=1, n_neighbors=9, p=2,
                     weights='uniform')
#predicts the test set results
y_pred = classifier.predict(x_test)

print(y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))