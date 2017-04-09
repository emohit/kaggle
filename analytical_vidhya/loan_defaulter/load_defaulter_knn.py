# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:46:11 2016

@author: monarang
"""


from sklearn import neighbors
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV


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

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
i_train = imp.fit_transform(train)

#converted ndarray back to dataframe as it is easy to manipulate
X_imputed_df = pd.DataFrame(i_train, columns = train.columns)

# build a classifier
clf = neighbors.KNeighborsClassifier(n_neighbors=2,weights='uniform')
#clf.fit(train,y)

tuned_parameters =  [{'n_neighbors': [2,5,8,10],'weights':['uniform' ,'distance']}]

gscv = GridSearchCV(clf, tuned_parameters, cv=15,verbose=5)
                       
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

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
i_test = imp.fit_transform(test)

#converted ndarray back to dataframe as it is easy to manipulate
test_imputed_df = pd.DataFrame(i_test, columns = test.columns)


predictions=gscv.predict(test_imputed_df)


submission = pd.DataFrame({ 'Loan_ID': test_data['Loan_ID'],
                            'Loan_Status': predictions })
submission.to_csv("C:\\Deloitte\\Kaggle\\analytical_vidhya\\loan_defaulter\\submission_knn.csv", index=False)