#Loading required Modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#Loading Datasets
iris = datasets.load_iris()

#Printing Description
#print(iris.DESCR)

features = iris.data
labels = iris.target
#print(features[0], labels[0])

#Training Classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

#Prediction
preds = clf.predict([[5.1, 3.5, 1.4, 0.2]])
print(preds)