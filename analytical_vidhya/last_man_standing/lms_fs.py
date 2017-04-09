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
from sklearn.feature_selection import SelectKBest,chi2,f_classif


root_path = "C:\\Deloitte\\Kaggle\\analytical_vidhya\\\last_man_standing\\"
train_data = pd.read_csv(root_path+"Train_Fyxd0t8.csv",)
test_data = pd.read_csv(root_path+"Test_C1XBIYq.csv")

train = train_data.drop(['Crop_Damage','ID'], axis=1)
test = test_data.drop(['ID'], axis=1)
y = train_data.Crop_Damage

imp = Imputer(missing_values='NaN', strategy='mean', axis=0,verbose=5)
i_train_1 = imp.fit(pd.concat([train,test]))

i_train = imp.transform(train)
#converted ndarray back to dataframe as it is easy to manipulate
X_imputed_df = pd.DataFrame(i_train, columns = train.columns)

X_imputed_df['no_doses_mul']=X_imputed_df['Number_Doses_Week']*X_imputed_df['Number_Weeks_Used']
X_imputed_df['no_doses_add']=X_imputed_df['Number_Doses_Week']+X_imputed_df['Number_Weeks_Used']
X_imputed_df['no_doses_sub']=abs(X_imputed_df['Number_Doses_Week']-X_imputed_df['Number_Weeks_Used'])

X_imputed_df['no_doses_mul1']=X_imputed_df['Number_Weeks_Used']*X_imputed_df['Number_Weeks_Quit']
X_imputed_df['no_doses_add1']=X_imputed_df['Number_Weeks_Used']+X_imputed_df['Number_Weeks_Quit']
X_imputed_df['no_doses_sub1']=X_imputed_df['Estimated_Insects_Count']/X_imputed_df['Pesticide_Use_Category']


##clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),('classification', svm.SVC())])
#clf = RandomForestClassifier(n_estimators=20)
#svr = svm.SVC()
#etc = AdaBoostClassifier(n_estimators=100,learning_rate=0.1, random_state=0)
#gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0)
#parameters = {'kernel':['rbf','poly'], 'C':[0.01,0.05,0.1,0.5], 'gamma': [0.1,0.01,1e-3]}
#
#parameters_gbm = {'n_estimators':[10,30,50],'learning_rate':[0.01,0.1,0.5,1.0],'max_depth':[3]}
#
#parameters_etc = {'n_estimators':[50,100,150],'learning_rate':[0.01,0.1,0.5,1.0]}
#
#parameters_clf = {'n_estimators':[30,50,100]}
#
##train['nan_count']=train.isnull().sum(axis=1)
#
##y=le.fit_transform(y)
##i_train=preprocessing.StandardScaler().fit_transform(i_train)
##X_imputed_df = X_imputed_df.drop(['Education','Self_Employed','CoapplicantIncome','LoanAmount','Property_Area','bol_dependent','Dependents','Married',u'Gender', u'Loan_Amount_Term'], axis=1)==
#
#
#gscv = GridSearchCV(gbm, parameters_gbm,verbose=5)
#                       
#gscv.fit(X_imputed_df, y)
#
#print "column used : " + str(X_imputed_df.columns)
#print "Best estimator : "+ str(gscv.best_estimator_)
#print "Best score of the estimator : "+  str(gscv.best_score_)
#

selector_chi = SelectKBest(chi2,4)
selector_chi.fit(X_imputed_df,y)

selector_f = SelectKBest(f_classif,4)
selector_f.fit(X_imputed_df,y)

print selector_chi.pvalues_
print selector_f.pvalues_
print X_imputed_df.columns
