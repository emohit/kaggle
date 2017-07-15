# -*- coding: utf-8 -*-
"""
Created on Wed May 03 20:57:12 2017

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
import matplotlib.pyplot as plt
 
root_path = "C:\\Users\\monarang\\Documents\\Projects\\Kaggle\\deloitte\\data_hackthon_2017\\"
train_data = pd.read_csv(root_path+"HackathonRound1.csv")
update_train_data = pd.read_csv(root_path+"DataUpdate_Hackathon.csv")

train_data = pd.concat([train_data,update_train_data])
train_data.to_csv(root_path+'concat_data.csv', index=False,header=True)

#split data in 2 seperate file one for open and one for close

open_price = pd.DataFrame({ 'Date': train_data['Date'], 'Share': train_data['Share Names'],
                            'Open': train_data['Open Price'] })
open_price.to_csv(root_path+"open.csv", index=False)

close_price = pd.DataFrame({ 'Date': train_data['Date'], 'Share': train_data['Share Names'],
                            'close': train_data['Close Price'] })
close_price.to_csv(root_path+"close.csv", index=False)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data_open = pd.read_csv(root_path+"open.csv",  index_col='Date',parse_dates=[0])
data_close = pd.read_csv(root_path+"close.csv",  index_col='Date',parse_dates=[0])

#print data.head()
#data
#model = ARIMA(data['Open'], order=(5,1,0))
#model_fit = model.fit(disp=0)
#print(model_fit.summary())

#data.drop(['Share'],axis=1,inplace=True)
#data=data.values

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def create_seperate_file(data,column):
    for i in range(1,51):
        df=data[data['Share']=='Share'+str(i)][column]
        df.to_csv(root_path+column+'_Share'+str(i), index=False,header=False)
        
        
create_seperate_file(data_open,'Open')
create_seperate_file(data_close,'close')

def shift_save(file_name,window):
    shift_file = pd.read_csv(root_path+file_name,  index_col='Date',parse_dates=[0])
    
