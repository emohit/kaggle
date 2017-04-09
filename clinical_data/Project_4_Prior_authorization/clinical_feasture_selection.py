# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:46:11 2016

@author: monarang
"""
from sklearn.ensemble  import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.feature_selection import SelectKBest,chi2,f_classif
selector = SelectKBest(chi2,4)
from sklearn.feature_selection import RFECV
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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

selector_chi = SelectKBest(chi2,4)
selector_chi.fit(train,y)

selector_f = SelectKBest(f_classif,4)
selector_f.fit(train,y)

print selector_chi.pvalues_
print selector_f.pvalues_
print train.columns

selector = RFECV(gbm, step=1, cv=5,verbose=5)
selector1 = selector.fit(X_imputed_df, y)

print selector1.support_ 

print selector1.ranking_

a={'columns_name':X_imputed_df.columns,'support':selector1.support_ ,'ranking':selector1.ranking_}
print pd.DataFrame(data=a).sort('ranking')

X_imputed_df_1=X_test[[u'UserID', u'Drug', u'Drug_Chemical_Name', u'GPI', u'Drug_Full_GPI_Name', u'NDC', u'DoctorID', u'Bin', u'State']]

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
fitter = pca.fit(X_imputed_df_1)
X1 = pca.transform(X_imputed_df_1)