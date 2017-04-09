# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:46:11 2016

@author: monarang
"""
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn import svm
#from sklearn.feature_selection import SelectKModel,f_classif
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM

root_path = "C:\\Deloitte\\Kaggle\\clinical_data\\Project_4_Prior_authorization\\"
train_data = pd.read_csv(root_path+"Predict _Prior auth_Data.csv",)

train = train_data.drop(['Target'], axis=1)
y = train_data.Target
date_time=pd.to_datetime(train_data.TransDate,format='%m/%d/%Y')
train['day']=date_time.apply(lambda x: x.day)

train['month']=date_time.apply(lambda x: x.month)

train['year']=date_time.apply(lambda x: x.year)


le = preprocessing.LabelEncoder()

train.UserID = le.fit_transform(train.UserID)
train.Drug = le.fit_transform(train.Drug)
train.DrugSubClass = le.fit_transform(train.DrugSubClass)
train.DrugClass = le.fit_transform(train.DrugClass)
train.Drug_Chemical_Name = le.fit_transform(train.Drug_Chemical_Name)
train.GPI = le.fit_transform(train.GPI)
train.Drug_Full_GPI_Name = le.fit_transform(train.Drug_Full_GPI_Name)
train.NDC = le.fit_transform(train.NDC)
train.DrugGroup = le.fit_transform(train.DrugGroup)
train.DoctorID = le.fit_transform(train.DoctorID)
train.RxGroupId = le.fit_transform(train.RxGroupId)
train.Bin = le.fit_transform(train.Bin)
train.PCN = le.fit_transform(train.PCN)
train.Zip = le.fit_transform(train.Zip)
train.State = le.fit_transform(train.State)
train.day = le.fit_transform(train.day)
train.month = le.fit_transform(train.month)
train.year = le.fit_transform(train.year)

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=0)
    
    
#clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),('classification', svm.SVC())])
clf = RandomForestClassifier(n_estimators=20)
svr = svm.SVC()
etc = AdaBoostClassifier(n_estimators=100,learning_rate=0.1, random_state=0)
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0)
knn = KNeighborsClassifier()
rbm = BernoulliRBM(random_state=0)

parameters = {'kernel':['poly'], 'C':[0.01,0.05,0.1,0.5], 'gamma': [0.1,0.01,1e-3]}

parameters_gbm = {'n_estimators':[50,80,100],'learning_rate':[0.01,0.1,0.5,1.0],'max_depth':[3]}

parameters_etc = {'n_estimators':[50,100,150],'learning_rate':[0.01,0.1,0.5,1.0]}

parameters_clf = {'n_estimators':[30,50,100]}

parameters_knn = {'n_neighbors':[3,4,5,6,7,8,9,10],'weights':['uniform','distance'],'algorithm':['auto','ball_tree','kd_tree','brute']}

parameters_rbm = {'learning_rate':[0.01,0.1,0.5,1.0]}

                     
X_train=X_train.drop(['TransDate',u'UserID', u'Drug', u'Drug_Chemical_Name', u'GPI', u'Drug_Full_GPI_Name', u'NDC', u'DoctorID', u'Bin', u'State'], axis=1)
                       
#X_train.DrugSubClass=preprocessing.StandardScaler().fit_transform(X_train.DrugSubClass)
#X_train.DrugClass=preprocessing.StandardScaler().fit_transform(X_train.DrugClass)
#X_train.DrugGroup=preprocessing.StandardScaler().fit_transform(X_train.DrugGroup)
#X_train.RxGroupId=preprocessing.StandardScaler().fit_transform(X_train.RxGroupId)
#X_train.PCN=preprocessing.StandardScaler().fit_transform(X_train.PCN)
#X_train.Zip=preprocessing.StandardScaler().fit_transform(X_train.Zip)
#X_train.day=preprocessing.StandardScaler().fit_transform(X_train.day)
#X_train.month=preprocessing.StandardScaler().fit_transform(X_train.month)
#X_train.year=preprocessing.StandardScaler().fit_transform(X_train.year)

     
gscv = GridSearchCV(rbm, parameters_rbm, cv=30,verbose=5)
                    
gscv.fit(X_train, y_train)

print "column used : " + str(X_train.columns)
print "Best estimator : "+ str(gscv.best_estimator_)
print "Best score of the estimator : "+  str(gscv.best_score_)

X_test=X_test.drop(['TransDate',u'UserID', u'Drug', u'Drug_Chemical_Name', u'GPI', u'Drug_Full_GPI_Name', u'NDC', u'DoctorID', u'Bin', u'State'], axis=1)

predictions=gscv.predict(X_test)
print predictions

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
