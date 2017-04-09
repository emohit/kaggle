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


root_path = "C:\\Deloitte\\Kaggle\\analytical_vidhya\\loan_defaulter\\"
train_data = pd.read_csv(root_path+"train_u6lujuX.csv",)
test_data = pd.read_csv(root_path+"test_Y3wMUE5.csv")

train = train_data.drop(['Loan_Status','Loan_ID'], axis=1)
test = test_data.drop(['Loan_ID'], axis=1)
y = train_data.Loan_Status

le = preprocessing.LabelEncoder()

#new feature



train.Dependents=train.Dependents.replace('3+',3)

#not imputing for the column 
train['Credit_History']=np.where(np.isnan(train_data['Credit_History']),99,train_data['Credit_History'])
#train['Dependents']=np.where(train['Dependents']=='nan',99,train['Dependents'])
#train['nan_LoanAmount']=np.where(np.isnan(train_data['LoanAmount']),99,1)
#train['Self_Employed']=np.where(pd.isnull(train_data['Self_Employed']),'OK',train_data['Self_Employed'])


train.Property_Area = le.fit_transform(train.Property_Area)
train.Gender = le.fit_transform(train.Gender)
train.Married = le.fit_transform(train.Married)
train.Education = le.fit_transform(train.Education)
train.Self_Employed = le.fit_transform(train.Self_Employed)


test.Dependents=test.Dependents.replace('3+',3)

test.Property_Area = le.fit_transform(test.Property_Area)
test.Gender = le.fit_transform(test.Gender)
test.Married = le.fit_transform(test.Married)
test.Education = le.fit_transform(test.Education)
test.Self_Employed = le.fit_transform(test.Self_Employed)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0,verbose=5)
i_train_1 = imp.fit(pd.concat([train,test]))

i_train = imp.transform(train)
#converted ndarray back to dataframe as it is easy to manipulate
X_imputed_df = pd.DataFrame(i_train, columns = train.columns)

#clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),('classification', svm.SVC())])
clf = RandomForestClassifier(n_estimators=20)
svr = svm.SVC()
etc = AdaBoostClassifier(n_estimators=100,learning_rate=0.1, random_state=0)
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0)
parameters = {'kernel':['rbf','poly'], 'C':[0.01,0.05,0.1,0.5], 'gamma': [0.1,0.01,1e-3]}

parameters_gbm = {'n_estimators':[10,30,50],'learning_rate':[0.01,0.1,0.5,1.0],'max_depth':[1,2,3]}

parameters_etc = {'n_estimators':[50,100,150],'learning_rate':[0.01,0.1,0.5,1.0]}

parameters_clf = {'n_estimators':[30,50,100]}

#some new features
X_imputed_df['debt_ratio']=X_imputed_df.LoanAmount/X_imputed_df.ApplicantIncome
X_imputed_df['debt_co_ratio']=X_imputed_df.LoanAmount/(X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome)
X_imputed_df['sum_of_co_salary']=X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome
X_imputed_df['avg_of_co_salary']=(X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome)/2
X_imputed_df['diff_of_co_salary']=abs(X_imputed_df.ApplicantIncome-X_imputed_df.CoapplicantIncome)
X_imputed_df['special_calculation']=(X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome)/X_imputed_df.Loan_Amount_Term
#X_imputed_df['special_calculation_1']=(X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome)/X_imputed_df.Dependents.replace(0,1)

X_imputed_df['bol_dependent']=le.fit_transform(X_imputed_df.Dependents > 0)
#train['nan_count']=train.isnull().sum(axis=1)

#y=le.fit_transform(y)
#i_train=preprocessing.StandardScaler().fit_transform(i_train)
X_imputed_df = X_imputed_df.drop(['Education','Self_Employed','CoapplicantIncome','LoanAmount','Property_Area','bol_dependent','Dependents','Married',u'Gender', u'Loan_Amount_Term'], axis=1)


gscv = GridSearchCV(gbm, parameters_gbm,cv=50,verbose=5)
                       
gscv.fit(X_imputed_df, y)

print "column used : " + str(X_imputed_df.columns)
print "Best estimator : "+ str(gscv.best_estimator_)
print "Best score of the estimator : "+  str(gscv.best_score_)

predictions=gscv.predict(X_imputed_df)



#test.Dependents=test.Dependents.replace('3+',3)
#
#test.Property_Area = le.fit_transform(test.Property_Area)
#test.Gender = le.fit_transform(test.Gender)
#test.Married = le.fit_transform(test.Married)
#test.Education = le.fit_transform(test.Education)
#test.Self_Employed = le.fit_transform(test.Self_Employed)

#test['nan_count']=test.isnull().sum(axis=1)
#not imputing for the column 
#test['Credit_History']=np.where(np.isnan(test_data['Credit_History']),99,test_data['Credit_History'])
#test['Dependents']=np.where(test['Dependents']=='nan',99,test['Dependents'])
#test['nan_LoanAmount']=np.where(np.isnan(test_data['LoanAmount']),99,1)


#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
i_test = imp.transform(test)

#converted ndarray back to dataframe as it is easy to manipulate
test_imputed_df = pd.DataFrame(i_test, columns = test.columns)

test_imputed_df['debt_ratio']=test_imputed_df.LoanAmount/np.where(test_imputed_df.ApplicantIncome==0,test_imputed_df.CoapplicantIncome,test_imputed_df.ApplicantIncome)
test_imputed_df['debt_co_ratio']=test_imputed_df.LoanAmount/(test_imputed_df.ApplicantIncome+test_imputed_df.CoapplicantIncome)
test_imputed_df['sum_of_co_salary']=test_imputed_df.ApplicantIncome+test_imputed_df.CoapplicantIncome
test_imputed_df['avg_of_co_salary']=(test_imputed_df.ApplicantIncome+test_imputed_df.CoapplicantIncome)/2
test_imputed_df['diff_of_co_salary']=abs(test_imputed_df.ApplicantIncome-test_imputed_df.CoapplicantIncome)
test_imputed_df['special_calculation']=(test_imputed_df.ApplicantIncome+test_imputed_df.CoapplicantIncome)/test_imputed_df.Loan_Amount_Term.replace(0,360)
#X_imputed_df['special_calculation_1']=(X_imputed_df.ApplicantIncome+X_imputed_df.CoapplicantIncome)/X_imputed_df.Dependents.replace(0,1)
test_imputed_df['bol_dependent']=le.fit_transform(test_imputed_df.Dependents > 0)


test_imputed_df = test_imputed_df.drop(['Education','Self_Employed','CoapplicantIncome','ApplicantIncome','Property_Area','bol_dependent','Dependents','Married',u'Gender', u'Loan_Amount_Term'], axis=1)


predictions=gscv.predict(test_imputed_df)
print predictions

submission = pd.DataFrame({ 'Loan_ID': test_data['Loan_ID'],
                            'Loan_Status': predictions })
submission.to_csv("C:\\Deloitte\\Kaggle\\analytical_vidhya\\loan_defaulter\\submission_svm.csv", index=False)