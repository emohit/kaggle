# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:46:11 2016

@author: monarang
"""
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.feature_selection import SelectFromModel


root_path = "C:\\Deloitte\\Kaggle\\analytical_vidhya\\loan_defaulter\\"
train_data = pd.read_csv(root_path+"train_u6lujuX.csv",)
test_data = pd.read_csv(root_path+"test_Y3wMUE5.csv")

train = train_data.drop(['Loan_ID','Loan_Status'], axis=1)
test = test_data.drop('Loan_ID', axis=1)
y = train_data.Loan_Status

le = preprocessing.LabelEncoder()

#new feature
train['nan_count']=train.isnull().sum(axis=1)


train.Dependents=train.Dependents.replace('3+',3)

train.Property_Area = le.fit_transform(train.Property_Area)
train.Gender = le.fit_transform(train.Gender)
train.Married = le.fit_transform(train.Married)
train.Education = le.fit_transform(train.Education)
train.Self_Employed = le.fit_transform(train.Self_Employed)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
i_train = imp.fit_transform(train)

#converted ndarray back to dataframe as it is easy to manipulate
X_imputed_df = pd.DataFrame(i_train, columns = train.columns)

clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),('classification', svm.SVC())])

#svr = svm.SVC()
parameters = {'classification__kernel':['rbf'], 'classification__C':[0.1,0.5,1,5,10], 'classification__gamma': [1e-3, 1e-4, 1e-5]}

X_imputed_df.LoanAmount=preprocessing.StandardScaler().fit_transform(X_imputed_df.LoanAmount)
X_imputed_df.ApplicantIncome=preprocessing.StandardScaler().fit_transform(X_imputed_df.ApplicantIncome)
X_imputed_df.CoapplicantIncome=preprocessing.StandardScaler().fit_transform(X_imputed_df.CoapplicantIncome)
X_imputed_df.Loan_Amount_Term=preprocessing.StandardScaler().fit_transform(X_imputed_df.Loan_Amount_Term)

#some new features
train['bol_dependent']=le.fit_transform(X_imputed_df.Dependents > 0)

#y=le.fit_transform(y)
#i_train=preprocessing.StandardScaler().fit_transform(i_train)
gscv = GridSearchCV(clf, parameters, cv=15,verbose=5)
                       
gscv.fit(X_imputed_df, y)


print "Best estimator : "+ str(gscv.best_estimator_)
print "Best score of the estimator : "+  str(gscv.best_score_)




test.Dependents=test.Dependents.replace('3+',3)

test.Property_Area = le.fit_transform(test.Property_Area)
test.Gender = le.fit_transform(test.Gender)
test.Married = le.fit_transform(test.Married)
test.Education = le.fit_transform(test.Education)
test.Self_Employed = le.fit_transform(test.Self_Employed)

test['nan_count']=test.isnull().sum(axis=1)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
i_test = imp.fit_transform(test)

#converted ndarray back to dataframe as it is easy to manipulate
test_imputed_df = pd.DataFrame(i_test, columns = test.columns)


test_imputed_df.LoanAmount=preprocessing.StandardScaler().fit_transform(test_imputed_df.LoanAmount)
test_imputed_df.ApplicantIncome=preprocessing.StandardScaler().fit_transform(test_imputed_df.ApplicantIncome)
test_imputed_df.CoapplicantIncome=preprocessing.StandardScaler().fit_transform(test_imputed_df.CoapplicantIncome)
test_imputed_df.Loan_Amount_Term=preprocessing.StandardScaler().fit_transform(test_imputed_df.Loan_Amount_Term)

test['bol_dependent']=le.fit_transform(test_imputed_df.Dependents > 0)

predictions=gscv.predict(test_imputed_df)
print predictions

submission = pd.DataFrame({ 'Loan_ID': test_data['Loan_ID'],
                            'Loan_Status': predictions })
submission.to_csv("C:\\Deloitte\\Kaggle\\analytical_vidhya\\loan_defaulter\\submission_svm.csv", index=False)