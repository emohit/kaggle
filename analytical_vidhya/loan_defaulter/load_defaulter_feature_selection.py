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

root_path = "C:\\Deloitte\\Kaggle\\analytical_vidhya\\loan_defaulter\\"
train_data = pd.read_csv(root_path+"train_u6lujuX.csv",)
test_data = pd.read_csv(root_path+"test_Y3wMUE5.csv")

train = train_data.drop(['Loan_ID','Loan_Status'], axis=1)
test = test_data.drop(['Loan_ID',u'Gender', u'Married', u'Dependents', u'Self_Employed', u'ApplicantIncome', u'CoapplicantIncome', u'LoanAmount', u'Loan_Amount_Term', u'Property_Area'], axis=1)
y = train_data.Loan_Status

le = preprocessing.LabelEncoder()

#new feature
train['nan_count']=train.isnull().sum(axis=1)


#not imputing for the column 
#train['Credit_History']=np.where(np.isnan(train_data['Credit_History']),99,train_data['Credit_History'])
#train['Dependents']=np.where(train['Dependents']=='nan',99,train['Dependents'])
#train['nan_LoanAmount']=np.where(np.isnan(train_data['LoanAmount']),99,1)
#train['Self_Employed']=np.where(pd.isnull(train_data['Self_Employed']),'OK',train_data['Self_Employed'])


train.Dependents=train.Dependents.replace('3+',3)

train.Property_Area = le.fit_transform(train.Property_Area)
train.Gender = le.fit_transform(train.Gender)
train.Married = le.fit_transform(train.Married)
train.Education = le.fit_transform(train.Education)
train.Self_Employed = le.fit_transform(train.Self_Employed)

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
i_train = imp.fit_transform(train)

#converted ndarray back to dataframe as it is easy to manipulate
X_imputed_df = pd.DataFrame(i_train, columns = train.columns)

#some more new feature derive from existing one.
X_imputed_df['debt_ratio']=X_imputed_df.LoanAmount/X_imputed_df.ApplicantIncome
X_imputed_df['debt_co_ratio']=X_imputed_df.LoanAmount/(X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome)
X_imputed_df['sum_of_co_salary']=X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome
X_imputed_df['avg_of_co_salary']=(X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome)/2
X_imputed_df['bol_dependent']=le.fit_transform(X_imputed_df.Dependents > 0)
X_imputed_df['diff_of_co_salary']=abs(X_imputed_df.ApplicantIncome-X_imputed_df.CoapplicantIncome)
X_imputed_df['special_calculation']=(X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome)/X_imputed_df.Loan_Amount_Term
#X_imputed_df['special_calculation_1']=(X_imputed_df.ApplicantIncome)/X_imputed_df.Loan_Amount_Term

from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=30, learning_rate=0.5,max_depth=3, random_state=0)

parameters_gbm = {'n_estimators':[10,30,50],'learning_rate':[0.01,0.1,0.5,1.0],'max_depth':[1]}


selector_chi = SelectKBest(chi2,4)
selector_chi.fit(X_imputed_df,y)

selector_f = SelectKBest(f_classif,4)
selector_f.fit(X_imputed_df,y)

print selector_chi.pvalues_
print selector_f.pvalues_


selector = RFECV(gbm, step=1, cv=5,verbose=5)
selector1 = selector.fit(X_imputed_df, y)

print selector1.support_ 

print selector1.ranking_

a={'columns_name':X_imputed_df.columns,'support':selector1.support_ ,'ranking':selector1.ranking_}
print pd.DataFrame(data=a).sort('ranking')


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
fitter = pca.fit(X_imputed_df)
X = pca.transform(X_imputed_df)