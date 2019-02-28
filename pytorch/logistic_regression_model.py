import torch
import torch.nn.functional as F
from torch.autograd import Variable

class LogisticRegressionModel(torch.nn.Module):

  def __init__(self):
    super(LogisticRegressionModel, self).__init__()
    self.linear = torch.nn.Linear(1,1)

  def forward(self, x):
    return torch.nn.Sigmoid()(self.linear(x))

model = LogisticRegressionModel()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]])) 
y_data = Variable(torch.Tensor([[0.],[0.],[1.],[1.]]))

for epoch in range(1000):

  y_pred = model(x_data)
  loss = criterion(y_pred, y_data)
  print(epoch, loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


x = Variable(torch.Tensor([1.0]))
print("After Training", model(x).data[0].item() > 0.5)




  


