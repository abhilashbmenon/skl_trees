# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from pandas import read_csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
# load the iris datasets
data = read_csv('iots.csv', header=0)
dataset = data.values
# split data into X and y
X = dataset[:,0:5]
X_dict = X.to_dict().values()
vect = DictVectorizer(sparse=False)
X_vector = vect.fit_transform(X_dict)
Y = dataset[:,8]
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X_vector, Y)
print(model)
# make predictions
expected = Y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))