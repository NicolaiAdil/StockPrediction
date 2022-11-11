import pandas as pd
from pandas_datareader import data as pdr
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

#Yahoo finance API
import yfinance as yf

#import and scale the data
def import_data(csvFileName, stockName):
    scaler = StandardScaler()

    #Imports data from csv file
    # df = pd.read_csv(csvFileName) 

    #Import data directly from yahoo finance with yfinance
    yf.pdr_override()
    df = pdr.get_data_yahoo(stockName, period="max")

    dataframeNew = df.reset_index()['Close']
    dataframeNew = scaler.fit_transform(np.array(dataframeNew).reshape(-1,1)) #Squashes the values to [-1,1]

    return dataframeNew, scaler

#Create dataset for training and testing
def create_dataset(dataset,time_step=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    
    return np.array(dataX), np.array(dataY)

#Preapare the training and testing datasets
def prepare_dataset(csvFileName, steps, percent, stockName):
    dataset = import_data(csvFileName,stockName)[0]
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


#Here we train the model with tensorflow, fit the model to our data and evaluate the results
def train_model(csvFileName, steps, stockName, set_epochs=100, percent=80):
    #Training the model
    model = Sequential()

    model.add(LSTM(50,return_sequences=True, input_shape=(steps,1)))
    model.add(LSTM(50,dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(50,dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer ='rmsprop')

    #Fit the model to our training and testing data
    x_train, y_train, x_test, y_test = prepare_dataset(csvFileName, steps, percent, stockName)

    model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=set_epochs, batch_size=64, verbose=1)

    #Evaluating the model
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    math.sqrt(mean_squared_error(y_train,train_predict))
    math.sqrt(mean_squared_error(y_test,test_predict))

    return train_predict, test_predict, model, x_train, y_train

#Predicts future values using our model
def predict_future(csvFileName, steps, days_in_future, stockName, model, x_train, y_train, set_epochs=100, percent=80):
    y_future = [y_train[-1]]

    x_pred = x_train[-1:, :, :]
    y_pred = y_train[-1]

    #Continuosly predicts new values based on previous ones
    for i in range(days_in_future):
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1,1,1), axis=1)

        y_pred = model.predict(x_pred)

        y_future.append(y_pred[-1])

    y_future = np.array(y_future).reshape(-1,1)

    return  y_future

def plot_results(csvFileName, steps, days_in_future, stockName, set_epochs=100, percent=80):
    train_predict, test_predict, model, x_train, y_train = train_model(csvFileName, steps, stockName, set_epochs,percent)
    future_predict = predict_future(csvFileName, steps, days_in_future, stockName, model, x_train, y_train, set_epochs, percent)

    #Plotting the results of the model

    look_back_steps = steps

    #The oroginal training data
    dataframeNew,scaler = import_data(csvFileName, stockName)
    dataframeNew = scaler.inverse_transform(dataframeNew)

    #Training data
    trainPredictionPlot = np.empty_like(dataframeNew)
    trainPredictionPlot[:,:]=np.nan
    trainPredictionPlot[look_back_steps:len(train_predict)+look_back_steps,:]=train_predict
    trainPredictionPlot = scaler.inverse_transform(trainPredictionPlot)

    #Test data
    testPredictionPlot = np.empty_like(dataframeNew)
    testPredictionPlot[:,:] = np.nan
    testPredictionPlot[len(train_predict)+(look_back_steps*2)+1:len(dataframeNew)-1,:]=test_predict
    testPredictionPlot = scaler.inverse_transform(testPredictionPlot)

    #Future prediction data
    futurePredictionPlot = np.empty_like(dataframeNew)
    futurePredictionPlot[:,:] = np.nan
    futurePredictionPlot[len(train_predict)+look_back_steps:len(future_predict)+len(train_predict)+look_back_steps,:]=future_predict
    futurePredictionPlot = scaler.inverse_transform(futurePredictionPlot)

    #Creating the plots and formats back to stockprizes
    plt.plot(dataframeNew, label = "data") 
    plt.plot(trainPredictionPlot, label = "training data")
    plt.plot(testPredictionPlot, label = "test data")
    plt.plot(futurePredictionPlot, label = "prediction")
    plt.legend()
    plt.show()


def main():
    fileName = 'GJF.csv'
    steps = 50
    iterations = 3
    days_to_predict = 10
    percent_training_data = 80

    stockName1 = "GJF.OL"
    stockName2 = "STB.OL"

    plot_results(fileName, steps, days_to_predict, stockName1, iterations, percent_training_data)
    plot_results(fileName, steps, days_to_predict, stockName2, iterations, percent_training_data)

main()









    





 
