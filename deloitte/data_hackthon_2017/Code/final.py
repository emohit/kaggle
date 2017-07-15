# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:12:00 2017

@author: monarang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 03 20:57:12 2017

@author: monarang
"""

import pandas as pd
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
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
    


warnings.filterwarnings("ignore")

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

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



def read_and_model(file_name,window=10):
    root_path="C:\\Users\\monarang\\Documents\\Projects\\Kaggle\\deloitte\\data_hackthon_2017\\"
    X_train, y_train, X_test, y_test = load_data(root_path+file_name, window, True)
    X_train1, y_train1, X_test1, y_test1 = load_data(root_path+file_name, window, False)
    
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
submission.set_value([70,71],['Open.','Close'],str('0'))

submission.to_csv(root_path+'TECHNOSAURS_DATAHACKATHON.csv', index=False,header=True)

