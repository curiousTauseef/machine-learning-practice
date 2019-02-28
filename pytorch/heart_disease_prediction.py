import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd

#df = pd.read_csv('datasets/heart.csv')

#df = df.reindex(np.random.permutation(df.index))
#df.to_csv('datasets/heart.csv')

#y = df.target.values

#x_data = df.drop(['target'], axis = 1)

#x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

data = np.loadtxt('datasets/heart.csv', delimiter=',', skiprows=1)

#y_data = Variable(torch.Tensor(y)).float()
#x_data = Variable(torch.Tensor(x.values)).float()

y_data = Variable(torch.from_numpy(data[:,[-1]])).float()
x_data = Variable(torch.from_numpy(data[:, 0:13])).float()

x_train = x_data[:250]
y_train = y_data[:250]
y_train = y_train.view(-1,1)

x_test = x_data[250:]
y_test = y_data[250:]
y_test = y_test.view(-1,1)

class Model(torch.nn.Module):

  def __init__(self):

    super(Model, self).__init__()

    self.linear1 = torch.nn.Linear(13,8)
    self.linear2 = torch.nn.Linear(8,4)
    self.linear3 = torch.nn.Linear(4,2)
    self.linear4 = torch.nn.Linear(2,1)

    self.sigmoid = torch.nn.Sigmoid()
    

  def forward(self, x):
    y = self.sigmoid(self.linear1(x))
    y = self.sigmoid(self.linear2(y))
    y = self.sigmoid(self.linear3(y))
    y = self.sigmoid(self.linear4(y))
    return y


model = Model()

#Set loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10000):
  
  #Forward pass
  y_pred = model(x_train)

  #Calculate loss
  loss = criterion(y_pred, y_train)

  #Back Propogation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  #Metrics
  output = (y_pred>0.5).float()
  accuracy = ((output == y_train).float().sum())/y_train.shape[0]
  print(epoch, loss.item(), accuracy.item()*100)
  

#After Training

with torch.no_grad():
  output = model(x_test)
  output = (output>0.5).float()
  accuracy = (((output == y_test).float().sum())/y_test.shape[0])*100
  print("Accuracy: ", accuracy.item())
  



