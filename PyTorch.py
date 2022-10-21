#/ LIBRARIES /#
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib as plt

#/ LOCAL FILES /#
from Linear_Regression import getData

#Creates a linear regression class that inherits from the torch.nn.Module class
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


#Gets, and formats the data
csvFile = "GjF_OneYear.csv"
date, close = getData(csvFile)

close_train = np.array(close, dtype = np.float32)
close_train = close_train.reshape(-1,1) 

x_train = np.array(range(len(close_train)), dtype = np.float32)


#Creates the model
inputDim = len(close_train)
outputDim = len(close_train)
learningRate = 0.01
epochs = 15
model = linearRegression(inputDim, outputDim)

#Uses GPU if possible
if torch.cuda.is_available():
    model.cuda()



#Initializes the loss function and the optimizer
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)



#Train the model
for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(close_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(close_train))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))




#Test the model
with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)


#Show the function
plt.clf()
plt.plot(x_train, close_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
