import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Read the CSV file
data = pd.read_csv('iots_skl.csv')
# # List of all classes
# data['Class Label']
# List of unique classes
data['decision'].unique()
# Number of entries for each unique classes
class_group = data.groupby('decision').apply(lambda x: len(x))
class_group
# Plot bar chart based on Class Label
class_group.plot(kind='bar', grid=False)

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

cols_to_retain = ['whof', 'purposef', 'whatf', 'cstore', 'caction']

X_feature = data[cols_to_retain]
X_dict = X_feature.T.to_dict().values()

# turn list of dicts into a numpy array
vect = DictVectorizer(sparse=False)
X_vector = vect.fit_transform(X_dict)

# print the features
# vect.get_feature_names()

# 0 to 14 is train set
X_Train = X_vector
# 15th is test set
# X_Test = X_vector[-1:] 

# Used to vectorize the class label
le = LabelEncoder()
y_train = le.fit_transform(data['decision'])

from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_Train,y_train)
# Predict the test data, not seen earlier
le.inverse_transform(clf.predict(X_Train))

# prediction with the same training set
Train_predict = clf.predict(X_Train)

# The model predicted the training set correctly
(Train_predict == y_train).all()

# Metrics related to the DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
print ('Accuracy is:', accuracy_score(y_train, Train_predict))
print (classification_report(y_train, Train_predict))