from time import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


#Import data fra CSV
def import_csv(csvFileName):
    df = pd.read_csv(csvFileName)

    # print(df.head(12)) 

    dataframeNew = df.reset_index()['Close']

    # dataframeNew.shape
    # print(dataframeNew)

    return dataframeNew

#Scale the data
def scale_data(csvFileName):
    scaler = StandardScaler()

    dataframeNew = import_csv(csvFileName)

    dataframeNew = scaler.fit_transform(np.array(dataframeNew).reshape(-1,1)) #Squashes the values to [-1,1]

    return dataframeNew

#Present the data as plots
def show_data(csvFileName):
    data = import_csv(csvFileName)
    data_scaled = scale_data(csvFileName)
    
    plt.plot(data)
    plt.plot(data_scaled)

    plt.show()

# show_data('GjF_OneYear.csv')



#Create dataset for training and testing
def create_dataset(dataset,time_step=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    
    return np.array(dataX), np.array(dataY)

#Preapare the training and testing datasets
def prepare_dataset(csvFileName):
    #Separate training dataset and testing dataset
    train_size = int(len(scale_data(csvFileName))*0.8)
    test_size = len(scale_data(csvFileName))-train_size

    training_data = scale_data(csvFileName)[0:train_size,:]
    testing_data = scale_data(csvFileName)[train_size:len(scale_data(csvFileName)),:1]

    #Create the datasets
    timesteps = 100

    #Training data
    x_train, y_train = create_dataset(training_data,timesteps)
    #Testing data
    x_test, y_test = create_dataset(testing_data,timesteps)

    return x_train, y_train, x_test, y_test


    





 
