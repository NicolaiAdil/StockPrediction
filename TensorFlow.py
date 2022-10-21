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

def train_model(csvFileName, steps,epochs=100):
    #Training the model

    model = Sequential()

    model.add(LSTM(50,return_sequences=True, input_shape=(steps,1)))
    model.add(LSTM(50,dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(50,dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer ='rmsprop')

    # model.summary()

#     return model

# #Fit the model to our training and testing data
# def fit_model(csvFileName):
    x_train, y_train, x_test, y_test = prepare_dataset(csvFileName, steps, 80)
    # model = train_model()

    model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=200, batch_size=64, verbose=1)
#     return model

# #Evaluating the model
# def evaluate_model(csvFileName):
    # x_train, y_train, x_test, y_test = prepare_dataset(csvFileName)
    # model = fit_model(csvFileName)

    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    math.sqrt(mean_squared_error(y_train,train_predict))
    math.sqrt(mean_squared_error(y_test,test_predict))

    #Plotting the results of the model

    look_back_steps = steps

    dataset = scale_data(csvFileName) #The original stock-prize dataset

    trainPredictionPlot = np.empty_like(dataset)
    trainPredictionPlot[:,:]=np.nan
    trainPredictionPlot[look_back_steps:len(train_predict)+look_back_steps,:]=train_predict

    testPredictionPlot = np.empty_like(dataset)
    testPredictionPlot[:,:] = np.nan
    testPredictionPlot[len(train_predict)+(look_back_steps*2)+1:len(dataset)-1,:]=test_predict

    #Creating the plots
    plt.plot(scale_data(csvFileName))
    plt.plot(trainPredictionPlot)
    plt.plot(testPredictionPlot)
    plt.show()

train_model('GjF_OneYear.csv', 20, 200)









    





 
