import torch
from torch.autograd import Variable

# Pytorch Rythm 1 - Set up model using NN class

class LinearRegressionModel(torch.nn.Module):

  def __init__(self):
    super(LinearRegressionModel, self).__init__()
    self.linear = torch.nn.Linear(1, 1)
  
  def forward(self, x):
    return self.linear(x)
  
model = LinearRegressionModel()

# Pytorch Rythm 2 - Create the loss and optimizer 

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Pytroch Rythm 3 - Forward pass, Backward propogation, Update

x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]])) 
y_data = Variable(torch.Tensor([[2.0],[4.0],[6.0],[8.0]]))

for epoch in range(150):

  y_pred = model(x_data) #forward

  loss = criterion(y_pred, y_data)

  #print(epoch, loss.item())

  optimizer.zero_grad()
  loss.backward()
  optimizer.step() #update weights

print("After Training Prediction (5):", model.forward(Variable(torch.Tensor([[5]]))))



