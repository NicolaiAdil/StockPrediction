import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#Tensorflow-packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM



#Import data fra CSV
def import_csv(csvFileName):
    df = pd.read_csv(csvFileName) 

    dataframeNew = df.reset_index()['Close']

    return dataframeNew

#Scale the data
def scale_data(csvFileName):
    scaler = StandardScaler()

    dataframeNew = import_csv(csvFileName)

    dataframeNew = scaler.fit_transform(np.array(dataframeNew).reshape(-1,1)) #Squashes the values to [-1,1]

    return dataframeNew

#Create dataset for training and testing
def create_dataset(dataset,time_step=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    
    return np.array(dataX), np.array(dataY)

#Preapare the training and testing datasets
def prepare_dataset(csvFileName, steps, percent):
    dataset = scale_data(csvFileName)
    #Separate training dataset and testing dataset
    train_size = int(len(dataset)*percent/100)
    test_size = len(dataset)-train_size

    training_data = dataset[0:train_size,:]
    testing_data = dataset[train_size:len(dataset),:1]

    #Create the datasets
    timesteps = steps

    #Training data
    x_train, y_train = create_dataset(training_data,timesteps)
    #Testing data
    x_test, y_test = create_dataset(testing_data,timesteps)

    #Reshaping
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)

    return x_train, y_train, x_test, y_test


#Here we train the model with tensorflow, fit the model to our data and then evaluate the results with plots
def train_model(csvFileName, steps, set_epochs=100, percent=80):
    #Training the model

    model = Sequential()

    model.add(LSTM(50,return_sequences=True, input_shape=(steps,1)))
    model.add(LSTM(50,dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(50,dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer ='rmsprop')

    #Fit the model to our training and testing data
    x_train, y_train, x_test, y_test = prepare_dataset(csvFileName, steps, percent)

    model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=set_epochs, batch_size=64, verbose=1)

    #Evaluating the model
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    math.sqrt(mean_squared_error(y_train,train_predict))
    math.sqrt(mean_squared_error(y_test,test_predict))

    #Plotting the results of the model

    look_back_steps = steps

    #Reformat the orginial training data
    dataframeNew = import_csv(csvFileName)
    scaler = StandardScaler()
    dataframeNew = scaler.fit_transform(np.array(dataframeNew).reshape(-1,1)) #Squashes the values to [-1,1]

    trainPredictionPlot = np.empty_like(dataframeNew)
    trainPredictionPlot[:,:]=np.nan
    trainPredictionPlot[look_back_steps:len(train_predict)+look_back_steps,:]=train_predict

    testPredictionPlot = np.empty_like(dataframeNew)
    testPredictionPlot[:,:] = np.nan
    testPredictionPlot[len(train_predict)+(look_back_steps*2)+1:len(dataframeNew)-1,:]=test_predict

    #Creating the plots and formats back to stockprizes
    plt.plot(scaler.inverse_transform(dataframeNew)) 
    plt.plot(scaler.inverse_transform(trainPredictionPlot))
    plt.plot(scaler.inverse_transform(testPredictionPlot))
    plt.show()

file = 'GJF.csv'
steps = 20
epochs = 20
percent_training_data = 75

train_model(file, steps, epochs, percent_training_data)









    





 
