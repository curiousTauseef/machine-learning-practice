import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import numpy as np
import pandas as pd

class HeartDiseaseDataset(Dataset):

  def __init__(self):
    
    data = np.loadtxt('datasets/heart.csv', delimiter=',', skiprows=1)
    self.y_data = torch.from_numpy(data[:,[-1]])
    self.x_data = torch.from_numpy(data[:, 0:13])

    self.y_data = self.y_data.view(-1,1)

    self.len = len(data)

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.len

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


dataset = HeartDiseaseDataset()

train_size = int(0.80*len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)

model = Model()

#Set loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
  
  for i, (x,y) in enumerate(train_loader):
    x,y = Variable(x).float(), Variable(y).float()
    #Forward pass
    y_pred = model(x)

    #Calculate loss
    loss = criterion(y_pred, y)

    #Back Propogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Metrics
    output = (y_pred>0.5).float()
    accuracy = ((output == y).float().sum())/y.shape[0]
    
print(epoch, i, loss.item(), accuracy.item()*100)

#After Training

with torch.no_grad():
  x,y = next(iter(test_loader))
  x,y = Variable(x).float(), Variable(y).float()
  output = model(x)
  output = (output>0.5).float()
  accuracy = (((output == y).float().sum())/y.shape[0])*100
  print("Accuracy on test data: ", accuracy.item())
  



