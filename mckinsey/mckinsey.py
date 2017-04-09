# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:46:11 2016

@author: monarang
"""
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.ensemble  import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn import svm
#from sklearn.feature_selection import SelectKModel,f_classif
import numpy as np
from sklearn import cross_validation
from sklearn import metrics
import math

root_path = "C:\\Deloitte\\Kaggle\\mckinsey\\"
train_data = pd.read_csv(root_path+"Train.csv",)
test_data = pd.read_csv(root_path+"Test.csv")

y = train_data['Email_Status']
train = train_data.drop(['Email_ID','Email_Status'], axis=1)
test = test_data.drop(['Email_ID'], axis=1)


le = preprocessing.LabelEncoder()

train['Customer_Location'] = le.fit_transform(train['Customer_Location'])
test['Customer_Location'] = le.fit_transform(test['Customer_Location'])


#train = train.drop(['County'], axis=1)
print train.shape

# to check nulls
print train.apply(lambda x: sum(x.isnull()))

##Divide into test and train:
#train = bigdata.loc[bigdata['source']=="train"]
#test = bigdata.loc[bigdata['source']=="test"]

#train.to_csv(root_path+"train_round.csv",index=False)
#test.to_csv(root_path+"test_round.csv",index=False)

#test.drop(['source','FIPS','YPLL.Rate'],axis=1,inplace=True)
#train.drop(['source','FIPS','YPLL.Rate'],axis=1,inplace=True)
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0,verbose=5)
i_train_1 = imp.fit(pd.concat([train,test]))

i_train = imp.transform(train)
#converted ndarray back to dataframe as it is easy to manipulate
X_imputed_df = pd.DataFrame(i_train, columns = train.columns)
print X_imputed_df.apply(lambda x: sum(x.isnull()))

#drop_column=[ u'Total_Links',
# u'Subject_Hotness_Score',
# u'Email_Campaign_Type',
# u'Total_Images',
# u'Customer_Location',
#u'Email_Type',
# u'Time_Email_sent_Category',
# u'Email_Source_Type']
#X_imputed_df = X_imputed_df.drop(drop_column, axis=1)
X_imputed_df['Subject_Hotness_Score']=X_imputed_df['Subject_Hotness_Score'].apply(lambda x: round(x,0))
#X_imputed_df['ration_wc_tot_link']=X_imputed_df.Word_Count/X_imputed_df.Total_Links
#X_imputed_df['ration_wc_tot_link']=X_imputed_df['ration_wc_tot_link'].apply(lambda x : round(x,0))
#
#X_imputed_df['ration_hot_tot_link']=X_imputed_df.Subject_Hotness_Score/X_imputed_df.Total_Links
#X_imputed_df['ration_hot_tot_link']=X_imputed_df['ration_hot_tot_link'].apply(lambda x : round(x,0))

#clf = randomforestregressor(n_estimators=20)
svr = svm.SVC()
gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,max_depth=3, random_state=0)
parameters = {'kernel':['rbf','poly'], 'C':[0.01,0.05,0.1,0.5], 'gamma': [0.1,0.01,1e-3]}

parameters_gbm = {'n_estimators':[300],'learning_rate':[0.1],'max_depth':[5,7]}

parameters_etc = {'n_estimators':[100,150],'learning_rate':[0.01,0.1,0.5,1.0]}

parameters_clf = {'n_estimators':[30,50,100]}

#fi = gbm.fit(train, y)
#features=pd.Series(fi.feature_importances_, train.columns).sort_values(ascending=False)[1:30]

#drop_columns = ['State','Perc.Obese','Perc.No.Soc.Emo.Support','Perc.Excessive.Drinking','Perc.pop.in.viol','County','Physically.Unhealthy.Days','Perc.Diabetic.Screening','Dentist.Ratio','Rec.Facility.Rate','Perc.High.School.Grad','Perc.Fair.Poor.Health','Pr.Care.Physician.Ratio','Avg.Daily.Particulates','Mentally.Unhealthy.Days','Perc.Mammography']
#drop_columns = ['Perc.Some.College','Perc.Fast.Foods','Chlamydia.Rate','Prev.Hosp.Stay.Rate','Violent.Crime.Rate','Perc.Limited.Access','Perc.Unemployed','Perc.Single.Parent.HH','Perc.Low.Birth.Weight','Perc.Smokers','Perc.Physically.Inactive','Teen.Birth.Rate','Perc.Uninsured','Perc.Children.in.Poverty','MV.Mortality.Rate']
#train = train.drop(drop_columns, axis=1)

gscv = GridSearchCV(gbm, parameters_gbm,cv=3,verbose=5)
                       
gscv.fit(X_imputed_df, y)

print "column used : " + str(train.columns)
print "Best estimator : "+ str(gscv.best_estimator_)
print "Best score of the estimator : "+  str(gscv.best_score_)
#print "Need less than 1319 RMSE :" + math.sqrt(-(gscv.best_score_))

#predictions=gscv.predict(train)

print test.apply(lambda x: sum(x.isnull()))

i_test = imp.transform(test)
#converted ndarray back to dataframe as it is easy to manipulate
X_imputed_df = pd.DataFrame(i_test, columns = test.columns)
print X_imputed_df.apply(lambda x: sum(x.isnull()))

#X_imputed_df = X_imputed_df.drop(drop_column, axis=1)
X_imputed_df['Subject_Hotness_Score']=X_imputed_df['Subject_Hotness_Score'].apply(lambda x: round(x,0))
#X_imputed_df['ration_wc_tot_link']=X_imputed_df.Word_Count/X_imputed_df.Total_Links
#X_imputed_df['ration_wc_tot_link']=X_imputed_df['ration_wc_tot_link'].apply(lambda x : round(x,0))
#
#X_imputed_df['ration_hot_tot_link']=X_imputed_df.Subject_Hotness_Score/X_imputed_df.Total_Links
#X_imputed_df['ration_hot_tot_link']=X_imputed_df['ration_hot_tot_link'].apply(lambda x : round(x,0))


predictions=gscv.predict(X_imputed_df)
print predictions

submission = pd.DataFrame({ 'Email_ID': test_data['Email_ID'],
                            'Email_Status': predictions })
submission.to_csv(root_path+"submission_svm.csv", index=False)