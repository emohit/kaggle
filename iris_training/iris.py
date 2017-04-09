# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:16:51 2015

@author: monarang
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X = iris.data
y = iris.target

print iris.data.shape

print iris.target.shape

knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression()


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)
knn.fit(X_train,y_train)
y_predict_knn = knn.predict(X_test)

logreg.fit(X_train,y_train)
y_predict_log = logreg.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print confusion_matrix(y_test,y_predict_knn)
print confusion_matrix(y_test,y_predict_log)

print accuracy_score(y_test,y_predict_knn)

for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_predict_knn = knn.predict(X_test)
    print i,accuracy_score(y_test,y_predict_knn)
    
    