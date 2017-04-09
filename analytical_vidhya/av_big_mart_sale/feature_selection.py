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
from sklearn.feature_selection import SelectKBest,f_regression


root_path = "C:\\Deloitte\\Kaggle\\analytical_vidhya\\av_big_mart_sale\\"
train_data = pd.read_csv(root_path+"Train_UWu5bXk.csv",)
test_data = pd.read_csv(root_path+"Test_u94Q5KV.csv")

realData =train_data[['Item_Identifier','Item_Weight']].groupby('Item_Identifier').max()
realData = pd.Series(data=realData.Item_Weight.values, index=realData.index)
train_data['Item_Weight']=train_data.Item_Identifier.map(realData)

realData =test_data[['Item_Identifier','Item_Weight']].groupby('Item_Identifier').max()
realData = pd.Series(data=realData.Item_Weight.values, index=realData.index)
test_data['Item_Weight']=train_data.Item_Identifier.map(realData)


train = train_data.drop(['Item_Outlet_Sales'], axis=1)
test = test_data#.drop(['Item_Identifier'], axis=1)
y = train_data.Item_Outlet_Sales

train.Item_Fat_Content=train.Item_Fat_Content.apply(lambda x : 'Low Fat' if (x =='LG' or x=='Low Fat') else 'Regular')
test.Item_Fat_Content=test.Item_Fat_Content.apply(lambda x : 'Low Fat' if (x =='LG' or x=='Low Fat') else 'Regular')
#
#factor = 3
#train.Item_Weight=train.Item_Weight.round(factor)
#train.Item_Visibility=train.Item_Visibility.round(factor)
#train.Item_MRP=train.Item_MRP.round(factor)
#
#test.Item_Weight=train.Item_Weight.round(factor)
#test.Item_Visibility=train.Item_Visibility.round(factor)
#test.Item_MRP=train.Item_MRP.round(factor)


le = preprocessing.LabelEncoder()

train.Item_Identifier = le.fit_transform(train.Item_Identifier)
train.Item_Fat_Content = le.fit_transform(train.Item_Fat_Content)
train.Item_Type = le.fit_transform(train.Item_Type)
train.Outlet_Identifier = le.fit_transform(train.Outlet_Identifier)
train.Outlet_Establishment_Year = le.fit_transform(train.Outlet_Establishment_Year)
train.Outlet_Size = le.fit_transform(train.Outlet_Size)
train.Outlet_Location_Type = le.fit_transform(train.Outlet_Location_Type)
train.Outlet_Type = le.fit_transform(train.Outlet_Type)

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


selector_f = SelectKBest(f_regression,4)
selector_f.fit(X_imputed_df,y)

print selector_f.pvalues_
print X_imputed_df.columns
