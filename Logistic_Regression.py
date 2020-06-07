from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()

#print(iris.keys())
#print(iris['data'].shape)
#print(iris['target'])

#Train a logistic regression for virginica
X = iris["data"][:, 3:]
#print(iris['data'][0])
#print(X[0])
y = (iris['target'] == 2).astype(np.int)
#print(y)

#Train a logistic regression
clf = LogisticRegression()
clf.fit(X,y)
example = clf.predict([[2.6]])
print(example)

#Using matplotlib for visualization

X_new = np.linspace(0,3,1000).reshape(-1, 1)
y_prob = clf.predict_proba(X_new)
plt.plot(X_new, y_prob[:, 1], "g-", label="virginica")
plt.show()