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

root_path = "C:\\Deloitte\\Kaggle\\deloitte\\"
train_data = pd.read_csv(root_path+"Train.csv",)
test_data = pd.read_csv(root_path+"Test.csv")

y = train_data['YPLL.Rate']
train = train_data.drop(['ID'], axis=1)
test = test_data.drop(['ID'], axis=1)
train['source']='train'
test['source']='test'

bigdata = pd.concat([test, train], ignore_index=True)

bigdata['FIPS1'] = bigdata.FIPS.apply(lambda x : str(x)[:2])
bigdata['FIPS2'] = bigdata.FIPS.apply(lambda x : str(x)[2:])


le = preprocessing.LabelEncoder()

#try replacing the mean value to mode and average and some default value like -9
#check with the randomforest instead of gbm
#create feature with FIPS with first 2 and last 2
#rounding of the variables to 0 has helped in getting the score 1209

bigdata['Perc.Fair.Poor.Health']=bigdata[['State','Perc.Fair.Poor.Health']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Physically.Unhealthy.Days']=bigdata[['State','Physically.Unhealthy.Days']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Mentally.Unhealthy.Days']=bigdata[['State','Mentally.Unhealthy.Days']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Low.Birth.Weight']=bigdata[['State','Perc.Low.Birth.Weight']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Smokers']=bigdata[['State','Perc.Smokers']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Obese']=bigdata[['State','Perc.Obese']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Physically.Inactive']=bigdata[['State','Perc.Physically.Inactive']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Excessive.Drinking']=bigdata[['State','Perc.Excessive.Drinking']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['MV.Mortality.Rate']=bigdata[['State','MV.Mortality.Rate']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Chlamydia.Rate']=bigdata[['State','Chlamydia.Rate']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Teen.Birth.Rate']=bigdata[['State','Teen.Birth.Rate']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Uninsured']=bigdata[['State','Perc.Uninsured']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Pr.Care.Physician.Ratio']=bigdata[['State','Pr.Care.Physician.Ratio']].groupby('State').transform(lambda x: x.fillna(round(x.mean().fillna(0),0)))
bigdata['Dentist.Ratio']=bigdata[['State','Dentist.Ratio']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Prev.Hosp.Stay.Rate']=bigdata[['State','Prev.Hosp.Stay.Rate']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Diabetic.Screening']=bigdata[['State','Perc.Diabetic.Screening']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Mammography']=bigdata[['State','Perc.Mammography']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.High.School.Grad']=bigdata[['State','Perc.High.School.Grad']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Some.College']=bigdata[['State','Perc.Some.College']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Unemployed']=bigdata[['State','Perc.Unemployed']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Children.in.Poverty']=bigdata[['State','Perc.Children.in.Poverty']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.No.Soc.Emo.Support']=bigdata[['State','Perc.No.Soc.Emo.Support']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Single.Parent.HH']=bigdata[['State','Perc.Single.Parent.HH']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Violent.Crime.Rate']=bigdata[['State','Violent.Crime.Rate']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Avg.Daily.Particulates']=bigdata[['State','Avg.Daily.Particulates']].groupby('State').transform(lambda x: x.fillna(x.mean().fillna(0)))
bigdata['Perc.pop.in.viol']=bigdata[['State','Perc.pop.in.viol']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Rec.Facility.Rate']=bigdata[['State','Rec.Facility.Rate']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Limited.Access']=bigdata[['State','Perc.Limited.Access']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))
bigdata['Perc.Fast.Foods']=bigdata[['State','Perc.Fast.Foods']].groupby('State').transform(lambda x: x.fillna(int(x.mean().fillna(0))))

bigdata['State'] = le.fit_transform(bigdata['State'])
bigdata['County'] = le.fit_transform(bigdata['County'])

#rounding data to the nearest zero
bigdata['Perc.Fair.Poor.Health']=bigdata['Perc.Fair.Poor.Health'].apply(lambda x: round(x,0))
bigdata['Physically.Unhealthy.Days']=bigdata['Physically.Unhealthy.Days'].apply(lambda x: round(x,0))
bigdata['Mentally.Unhealthy.Days']=bigdata['Mentally.Unhealthy.Days'].apply(lambda x: round(x,0))
bigdata['Perc.Low.Birth.Weight']=bigdata['Perc.Low.Birth.Weight'].apply(lambda x: round(x,0))
bigdata['Perc.Smokers']=bigdata['Perc.Smokers'].apply(lambda x: round(x,0))
bigdata['Perc.Obese']=bigdata['Perc.Obese'].apply(lambda x: round(x,0))
bigdata['Perc.Physically.Inactive']=bigdata['Perc.Physically.Inactive'].apply(lambda x: round(x,0))
bigdata['Perc.Excessive.Drinking']=bigdata['Perc.Excessive.Drinking'].apply(lambda x: round(x,0))
bigdata['MV.Mortality.Rate']=bigdata['MV.Mortality.Rate'].apply(lambda x: round(x,0))
bigdata['Chlamydia.Rate']=bigdata['Chlamydia.Rate'].apply(lambda x: round(x,0))
bigdata['Teen.Birth.Rate']=bigdata['Teen.Birth.Rate'].apply(lambda x: round(x,0))
bigdata['Perc.Uninsured']=bigdata['Perc.Uninsured'].apply(lambda x: round(x,0))
bigdata['Pr.Care.Physician.Ratio']=bigdata['Pr.Care.Physician.Ratio'].apply(lambda x: round(x,0))
bigdata['Dentist.Ratio']=bigdata['Dentist.Ratio'].apply(lambda x: round(x,0))
bigdata['Prev.Hosp.Stay.Rate']=bigdata['Prev.Hosp.Stay.Rate'].apply(lambda x: round(x,0))
bigdata['Perc.Diabetic.Screening']=bigdata['Perc.Diabetic.Screening'].apply(lambda x: round(x,0))
bigdata['Perc.Mammography']=bigdata['Perc.Mammography'].apply(lambda x: round(x,0))
bigdata['Perc.High.School.Grad']=bigdata['Perc.High.School.Grad'].apply(lambda x: round(x,0))
bigdata['Perc.Some.College']=bigdata['Perc.Some.College'].apply(lambda x: round(x,0))
bigdata['Perc.Unemployed']=bigdata['Perc.Unemployed'].apply(lambda x: round(x,0))
bigdata['Perc.Children.in.Poverty']=bigdata['Perc.Children.in.Poverty'].apply(lambda x: round(x,0))
bigdata['Perc.No.Soc.Emo.Support']=bigdata['Perc.No.Soc.Emo.Support'].apply(lambda x: round(x,0))
bigdata['Perc.Single.Parent.HH']=bigdata['Perc.Single.Parent.HH'].apply(lambda x: round(x,0))
bigdata['Violent.Crime.Rate']=bigdata['Violent.Crime.Rate'].apply(lambda x: round(x,0))
bigdata['Avg.Daily.Particulates']=bigdata['Avg.Daily.Particulates'].apply(lambda x: round(x,0))
bigdata['Perc.pop.in.viol']=bigdata['Perc.pop.in.viol'].apply(lambda x: round(x,0))
bigdata['Rec.Facility.Rate']=bigdata['Rec.Facility.Rate'].apply(lambda x: round(x,0))
bigdata['Perc.Limited.Access']=bigdata['Perc.Limited.Access'].apply(lambda x: round(x,0))
bigdata['Perc.Fast.Foods']=bigdata['Perc.Fast.Foods'].apply(lambda x: round(x,0))
#End of rounding


#train = train.drop(['County'], axis=1)
print bigdata.shape,train.shape,test.shape

# to check nulls
print bigdata.apply(lambda x: sum(x.isnull()))

#Divide into test and train:
train = bigdata.loc[bigdata['source']=="train"]
test = bigdata.loc[bigdata['source']=="test"]

train.to_csv(root_path+"train_round.csv",index=False)
test.to_csv(root_path+"test_round.csv",index=False)

test.drop(['source','FIPS','YPLL.Rate'],axis=1,inplace=True)
train.drop(['source','FIPS','YPLL.Rate'],axis=1,inplace=True)

#clf = randomforestregressor(n_estimators=20)
svr = svm.SVC()
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0)
parameters = {'kernel':['rbf','poly'], 'C':[0.01,0.05,0.1,0.5], 'gamma': [0.1,0.01,1e-3]}

parameters_gbm = {'n_estimators':[100],'learning_rate':[0.01,0.1,0.5,1.0],'max_depth':[3]}

parameters_etc = {'n_estimators':[50,100,150],'learning_rate':[0.01,0.1,0.5,1.0]}

parameters_clf = {'n_estimators':[30,50,100]}

#fi = gbm.fit(train, y)
#features=pd.Series(fi.feature_importances_, train.columns).sort_values(ascending=False)[1:30]

#drop_columns = ['State','Perc.Obese','Perc.No.Soc.Emo.Support','Perc.Excessive.Drinking','Perc.pop.in.viol','County','Physically.Unhealthy.Days','Perc.Diabetic.Screening','Dentist.Ratio','Rec.Facility.Rate','Perc.High.School.Grad','Perc.Fair.Poor.Health','Pr.Care.Physician.Ratio','Avg.Daily.Particulates','Mentally.Unhealthy.Days','Perc.Mammography']
#drop_columns = ['Perc.Some.College','Perc.Fast.Foods','Chlamydia.Rate','Prev.Hosp.Stay.Rate','Violent.Crime.Rate','Perc.Limited.Access','Perc.Unemployed','Perc.Single.Parent.HH','Perc.Low.Birth.Weight','Perc.Smokers','Perc.Physically.Inactive','Teen.Birth.Rate','Perc.Uninsured','Perc.Children.in.Poverty','MV.Mortality.Rate']
#train = train.drop(drop_columns, axis=1)

gscv = GridSearchCV(gbm, parameters_gbm,cv=10,verbose=5,scoring='mean_squared_error')
                       
gscv.fit(train, y)

print "column used : " + str(train.columns)
print "Best estimator : "+ str(gscv.best_estimator_)
print "Best score of the estimator : "+  str(gscv.best_score_)
print "Need less than 1319 RMSE :" + math.sqrt(-(gscv.best_score_))

predictions=gscv.predict(train)

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    print target
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print "\nModel Report"
    print "RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    print "CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(root_path+filename, index=False)

target='YPLL.Rate'
predictors = [x for x in train.columns if x not in target+'ID'+'County']
IDcol = ['ID']

alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort(ascending=False)


predictions=gscv.predict(test)
print predictions

submission = pd.DataFrame({ 'ID': test_data['ID'],
                            'Predicted': predictions })
submission.to_csv(root_path+"submission_svm.csv", index=False)