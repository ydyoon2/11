"""
RNN LSTM GRU
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

route = "C:/Users/aswes/Desktop/2019/Practicum/Crude/"

df = pd.read_csv("%sCombined.csv"%route, index_col = 0, parse_dates=[0])

def split_data(data, split_date):
    return data[data.index <= split_date].copy(), \
           data[data.index >  split_date].copy()

date = '2017-12-31'

sc = MinMaxScaler()
scaled_df = sc.fit_transform(df)

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

seq_length = 10
x, y = sliding_windows(scaled_df, seq_length)

train_size = int(len(y) * 0.8)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))


#scaled_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns)
#
#train, test = split_data(scaled_df, date)
#
#
#plt.figure(figsize=(15,5))
#plt.title('scaled_data')
#plt.xlabel('time')
#plt.ylabel('positions')
#plt.plot(train.index,train.iloc[:,:])
#plt.plot(test.index,test.iloc[:,:])
#plt.axvline(x=date, color='k', linestyle='--')
##plt.legend(train.iloc[:,:], fancybox=True, framealpha=0, loc='best')
#plt.show()
#
#plt.figure(figsize=(15,5))
#plt.title('no_abrupt')
#plt.xlabel('time')
#plt.ylabel('no')
#plt.plot(train.index,train.iloc[:,-1])
#plt.plot(test.index,test.iloc[:,-1])
#plt.axvline(x=date, color='k', linestyle='--')
##plt.legend(train.iloc[:,:], fancybox=True, framealpha=0, loc='best')
#plt.show()
#
#
#X_train = train.iloc[:,:-1] 
#y_train = train.iloc[:,-1]
#X_test =  test.iloc[:,:-1]
#y_test =  test.iloc[:,-1]
#
#trainX = Variable(torch.Tensor(np.array(X_train)))
#trainY = Variable(torch.Tensor(np.array(y_train)))
#testX = Variable(torch.Tensor(np.array(X_test)))
#testY = Variable(torch.Tensor(np.array(y_test)))

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

num_epochs = 10000
learning_rate = 0.001

input_size = 71
hidden_size = 2
num_layers = 1

num_classes = 71

lstm = LSTM(num_classes, input_size, hidden_size, num_layers).cuda()

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot[:,-1])
plt.plot(data_predict[:,-1])
plt.suptitle('Time-Series Prediction')
plt.show()























































#
#class LSTM(nn.Module):
#
#    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
#                    num_layers=2):
#        super(LSTM, self).__init__()
#        self.input_dim = input_dim
#        self.hidden_dim = hidden_dim
#        self.batch_size = batch_size
#        self.num_layers = num_layers
#
#        # Define the LSTM layer
#        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
#
#        # Define the output layer
#        self.linear = nn.Linear(self.hidden_dim, output_dim)
#
#    def init_hidden(self):
#        # This is what we'll initialise our hidden state as
#        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
#                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
#
#    def forward(self, input):
#        # Forward pass through LSTM layer
#        # shape of lstm_out: [input_size, batch_size, hidden_dim]
#        # shape of self.hidden: (a, b), where a and b both 
#        # have shape (num_layers, batch_size, hidden_dim).
#        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
#        
#        # Only take the output from the final timetep
#        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
#        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
#        return y_pred.view(-1)
#
#model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)
#
#
#loss_fn = torch.nn.MSELoss(size_average=False)
#
#optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
######################
## Train model
######################
#
#hist = np.zeros(num_epochs)
#
#for t in range(num_epochs):
#    # Clear stored gradient
#    model.zero_grad()
#    
#    # Initialise hidden state
#    # Don't do this if you want your LSTM to be stateful
#    model.hidden = model.init_hidden()
#    
#    # Forward pass
#    y_pred = model(X_train)
#
#    loss = loss_fn(y_pred, y_train)
#    if t % 100 == 0:
#        print("Epoch ", t, "MSE: ", loss.item())
#    hist[t] = loss.item()
#
#    # Zero out gradient, else they will accumulate between epochs
#    optimiser.zero_grad()
#
#    # Backward pass
#    loss.backward()
#
#    # Update parameters
#    optimiser.step()
