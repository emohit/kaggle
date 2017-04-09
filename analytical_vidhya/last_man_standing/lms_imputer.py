# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:40:17 2016

@author: monarang
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:46:11 2016

@author: monarang
"""
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,GradientBoostingRegressor
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


root_path = "C:\\Deloitte\\Kaggle\\analytical_vidhya\\\last_man_standing\\"
train_data = pd.read_csv(root_path+"Train_Fyxd0t8.csv",)


test_data=train_data[train_data.Number_Weeks_Used.isnull()]
train_data=train_data[~train_data.Number_Weeks_Used.isnull()]


y = train_data.Number_Weeks_Used
train = train_data.drop(['Crop_Damage','ID','Number_Weeks_Used'], axis=1)
test = test_data.drop(['ID','Crop_Damage','Number_Weeks_Used'], axis=1)

#clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),('classification', svm.SVC())])
clf = RandomForestClassifier(n_estimators=20)
svr = svm.SVC()
etc = AdaBoostClassifier(n_estimators=100,learning_rate=0.1, random_state=0)
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0)
parameters = {'kernel':['rbf','poly'], 'C':[0.01,0.05,0.1,0.5], 'gamma': [0.1,0.01,1e-3]}

parameters_gbm = {'n_estimators':[30],'learning_rate':[0.5,1.0],'max_depth':[3]}

parameters_etc = {'n_estimators':[50,100,150],'learning_rate':[0.01,0.1,0.5,1.0]}

parameters_clf = {'n_estimators':[30,50,100]}

#train['nan_count']=train.isnull().sum(axis=1)

#y=le.fit_transform(y)

drop_columns = [u'Number_Weeks_Used', u'Number_Weeks_Quit']
#X_imputed_df = X_imputed_df.drop(drop_columns, axis=1)


gscv = GridSearchCV(gbm, parameters_gbm,verbose=5)
                       
gscv.fit(train, y)

print "column used : " + str(train.columns)
print "Best estimator : "+ str(gscv.best_estimator_)
print "Best score of the estimator : "+  str(gscv.best_score_)


predictions=gscv.predict(test)
print predictions

submission = pd.DataFrame({ 'ID': test_data['ID'],
                            'Crop_Damage': predictions })
#submission.to_csv(root_path+"submission.csv", index=False)
