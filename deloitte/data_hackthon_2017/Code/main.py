# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:55:15 2017

@author: monarang
"""

import sys
sys.path.append('C:\\Users\\monarang\\Documents\\Projects\\Kaggle\\deloitte\\data_hackthon_2017\\')
sys.path.append('C:\\Users\\monarang\\Documents\\Projects\\Kaggle\\deloitte\\data_hackthon_2017\\Code')

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper libraries
from numpy import newaxis
import numpy as np
import pandas as pd


def read_and_model(file_name,window=10):
    root_path="C:\\Users\\monarang\\Documents\\Projects\\Kaggle\\deloitte\\data_hackthon_2017\\"
    X_train, y_train, X_test, y_test = lstm.load_data(root_path+file_name, window, True)
    X_train1, y_train1, X_test1, y_test1 = lstm.load_data(root_path+file_name, window, False)
    
    #Step 2 Build Model
    model = Sequential()
    
    model.add(LSTM(
        input_dim=1,
        output_dim=50,
        return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))
    
    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('compilation time : ', time.time() - start)
    
    #Step 3 Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=1,
        validation_split=0.05)
    return model,X_test,X_test1


def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)//prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

#predictions = predict_sequences_multiple(model, X_test, 10, 10)
#lstm.plot_results_multiple(predictions, y_test, 10)



def predict_sequence(model, data, window_size=10, prediction_len=4):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    last_Set = len(data)-1
    curr_frame = data[last_Set]
    predicted = []
    for j in range(prediction_len):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def denormalize_result(data):
    denom = []
    for i in range(len(data)):
        denom.append(round((float(data[i])+1)*float(X_test1[len(X_test1)-1][0][0]),2))
    return denom


close_price_result = []

for i in (range(50)):
    print(str(i)+" is going on............")
    file_name = 'close_share'+str(i+1)
    model,X_test,X_test1 = read_and_model(file_name)
    pred = predict_sequence(model,X_test)
    denom = denormalize_result(pred)
    close_price_result.append(denom[-2:])

close_price_for_csv = [j for i in close_price_result for j in i]

open_price_result = []

for i in (range(50)):
    print(str(i)+" is going on............")
    file_name = 'open_share'+str(i+1)
    model,X_test,X_test1 = read_and_model(file_name)
    pred = predict_sequence(model,X_test)
    denom = denormalize_result(pred)
    open_price_result.append(denom[-2:])

open_price_for_csv = [j for i in open_price_result for j in i]


#Few hardcoding
predicted_date=['18-May-17','19-May-17']*50

share_name = []
for i in range(50):
    share_name.append('Share'+str(i+1))
    share_name.append('Share'+str(i+1))


submission = pd.DataFrame(
    {'Date': predicted_date,
     'Share.': share_name,
     'Open.': open_price_for_csv,
     'Close': close_price_for_csv
     },columns=['Date','Share.','Open.','Close'])

#set zero Share36
submission.set_value([70,71],['Open.','Close'],0)

submission.to_csv(root_path+'hackathon_Sample_Submission.csv', index=False,header=True)



