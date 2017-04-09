# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:46:11 2016

@author: monarang
"""
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
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
from sklearn.neighbors import KNeighborsRegressor

root_path = "C:\\Deloitte\\Kaggle\\analytical_vidhya\\av_big_mart_sale\\"
train_data = pd.read_csv(root_path+"Train_UWu5bXk.csv",)
test_data = pd.read_csv(root_path+"Test_u94Q5KV.csv")



train_data['Item_Weight'] =train_data[['Item_Identifier','Item_Weight']].groupby('Item_Identifier').transform(lambda x: x.fillna(x.mean()))
test_data['Item_Weight'] =test_data[['Item_Identifier','Item_Weight']].groupby('Item_Identifier').transform(lambda x: x.fillna(x.mean()))

train_data['Item_Visibility']=train_data[['Item_Identifier','Item_Visibility']].groupby('Item_Identifier').transform(lambda x: x.replace(0,x.mean()))
test_data['Item_Visibility']=test_data[['Item_Identifier','Item_Visibility']].groupby('Item_Identifier').transform(lambda x: x.replace(0,x.mean()))
#train_data=train_data[train_data.Item_Visibility<>0]
#test_data=test_data[test_data.Item_Visibility<>0]


train = train_data.drop(['Item_Outlet_Sales'], axis=1)
test = test_data.copy()
y = train_data.Item_Outlet_Sales

train.Item_Fat_Content=train.Item_Fat_Content.apply(lambda x : 'Low Fat' if (x =='LG' or x=='Low Fat') else 'Regular')
test.Item_Fat_Content=test.Item_Fat_Content.apply(lambda x : 'Low Fat' if (x =='LG' or x=='Low Fat') else 'Regular')

#new features
train['item_identifier_1']=train_data.Item_Identifier.apply(lambda x : x[3:])
train['item_identifier_2']=train_data.Item_Identifier.apply(lambda x : x[:3])

train['item_identifier_11']=train_data.Item_Identifier.apply(lambda x : x[:1])
train['item_identifier_12']=train_data.Item_Identifier.apply(lambda x : x[1:2])
train['item_identifier_13']=train_data.Item_Identifier.apply(lambda x : x[2:3])


train['Outlet_Identifier_1']=train_data.Outlet_Identifier.apply(lambda x : x[3:])
#train['Outlet_Identifier_2']=train_data.Outlet_Identifier.apply(lambda x : x[:3])


test['item_identifier_1']=test_data.Item_Identifier.apply(lambda x : x[3:])
test['item_identifier_2']=test_data.Item_Identifier.apply(lambda x : x[:3])

test['item_identifier_11']=test_data.Item_Identifier.apply(lambda x : x[:1])
test['item_identifier_12']=test_data.Item_Identifier.apply(lambda x : x[1:2])
test['item_identifier_13']=test_data.Item_Identifier.apply(lambda x : x[2:3])


test['Outlet_Identifier_1']=test_data.Outlet_Identifier.apply(lambda x : x[3:])
#test['Outlet_Identifier_2']=test_data.Outlet_Identifier.apply(lambda x : x[:3])


#factor = 3
#train.Item_Weight=train.Item_Weight.round(factor)
#train.Item_Visibility=train.Item_Visibility.round(factor)
#train.Item_MRP=train.Item_MRP.round(factor)
#
#test.Item_Weight=train.Item_Weight.round(factor)
#test.Item_Visibility=train.Item_Visibility.round(factor)
#test.Item_MRP=train.Item_MRP.round(factor)


le = preprocessing.LabelEncoder()

train.item_identifier_1 = le.fit_transform(train.item_identifier_1)
train.item_identifier_2 = le.fit_transform(train.item_identifier_2)

train.item_identifier_11 = le.fit_transform(train.item_identifier_11)
train.item_identifier_12 = le.fit_transform(train.item_identifier_12)
train.item_identifier_13 = le.fit_transform(train.item_identifier_13)

train.Outlet_Identifier_1 = le.fit_transform(train.Outlet_Identifier_1)
#train.Outlet_Identifier_2 = le.fit_transform(train.Outlet_Identifier_2)


train.Item_Identifier = le.fit_transform(train.Item_Identifier)
train.Item_Fat_Content = le.fit_transform(train.Item_Fat_Content)
train.Item_Type = le.fit_transform(train.Item_Type)
train.Outlet_Identifier = le.fit_transform(train.Outlet_Identifier)
train.Outlet_Establishment_Year = le.fit_transform(train.Outlet_Establishment_Year)
train.Outlet_Size = le.fit_transform(train.Outlet_Size)
train.Outlet_Location_Type = le.fit_transform(train.Outlet_Location_Type)
train.Outlet_Type = le.fit_transform(train.Outlet_Type)

test.item_identifier_1 = le.fit_transform(test.item_identifier_1)
test.item_identifier_2 = le.fit_transform(test.item_identifier_2)

test.item_identifier_11 = le.fit_transform(test.item_identifier_11)
test.item_identifier_12= le.fit_transform(test.item_identifier_12)
test.item_identifier_13 = le.fit_transform(test.item_identifier_13)

test.Outlet_Identifier_1 = le.fit_transform(test.Outlet_Identifier_1)
#test.Outlet_Identifier_2 = le.fit_transform(test.Outlet_Identifier_2)


test.Item_Identifier = le.fit_transform(test.Item_Identifier)
test.Item_Fat_Content = le.fit_transform(test.Item_Fat_Content)
test.Item_Type = le.fit_transform(test.Item_Type)
test.Outlet_Identifier = le.fit_transform(test.Outlet_Identifier)
test.Outlet_Establishment_Year = le.fit_transform(test.Outlet_Establishment_Year)
test.Outlet_Size = le.fit_transform(test.Outlet_Size)
test.Outlet_Location_Type = le.fit_transform(test.Outlet_Location_Type)
test.Outlet_Type = le.fit_transform(test.Outlet_Type)


imp = Imputer(missing_values='NaN', strategy='median', axis=0,verbose=5)
i_train_1 = imp.fit(pd.concat([train,test]))

i_train = imp.transform(train)


#converted ndarray back to dataframe as it is easy to manipulate
X_imputed_df = pd.DataFrame(i_train, columns = train.columns)

X_imputed_df['ratio_mrp_weight']=X_imputed_df.Item_MRP/X_imputed_df.Item_Weight
X_imputed_df['Item_Visibility_2']=X_imputed_df.Item_Visibility/X_imputed_df.Item_Visibility

#clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),('classification', svm.SVC())])
clf = RandomForestClassifier(n_estimators=20)
svr = svm.SVR()
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0)
knn = KNeighborsRegressor()

parameters = {'kernel':['rbf','poly'], 'C':[0.01,0.05,0.1,0.5], 'gamma': [0.1,0.01,1e-3]}

parameters_gbm = {'loss' : ['ls'] ,'n_estimators':[50],'learning_rate':[0.1],'max_depth':[1,3]}

parameters_knn = {'n_neighbors':[5,10,15],'weights':['distance']}

parameters_clf = {'n_estimators':[30,50,100]}

drop_column=['Item_Visibility','Item_Type','Item_Fat_Content','Item_Weight','Item_Visibility_2','ratio_mrp_weight','item_identifier_13','item_identifier_11','item_identifier_12','Outlet_Location_Type','Outlet_Identifier','Outlet_Size','Outlet_Establishment_Year']
X_imputed_df = X_imputed_df.drop(drop_column, axis=1)
#Item_Type

gscv = GridSearchCV(gbm, parameters_gbm,cv=30,verbose=5)
                       
gscv.fit(X_imputed_df, y)

print "column used : " + str(X_imputed_df.columns)
print "Best estimator : "+ str(gscv.best_estimator_)
print "Best score of the estimator : "+  str(gscv.best_score_)


#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
i_test = imp.transform(test)

#converted ndarray back to dataframe as it is easy to manipulate
test_imputed_df = pd.DataFrame(i_test, columns = test.columns)

test_imputed_df['ratio_mrp_weight']=test_imputed_df.Item_MRP/test_imputed_df.Item_Weight
test_imputed_df['Item_Visibility_2']=test_imputed_df.Item_Visibility*test_imputed_df.Item_Visibility

test_imputed_df = test_imputed_df.drop(drop_column, axis=1)

predictions=gscv.predict(test_imputed_df)
print predictions

submission = pd.DataFrame({ 'Item_Identifier': test_data['Item_Identifier'],
                            'Outlet_Identifier': test_data['Outlet_Identifier'], 
                            'Item_Outlet_Sales' : predictions
                            })
submission.to_csv(root_path+"submission_svm.csv", index=False)