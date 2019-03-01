import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

class NeuralNet(torch.nn.Module):
  def __init__(self):
    super(NeuralNet, self).__init__()

    self.linear1 = torch.nn.Linear(784, 520)
    self.linear2 = torch.nn.Linear(520, 320)
    self.linear3 = torch.nn.Linear(320, 240)
    self.linear4 = torch.nn.Linear(240, 120)
    self.linear5 = torch.nn.Linear(120, 10)

    self.relu = torch.nn.ReLU()
  
  def forward(self, x): 
    x = x.view(-1,784)
    x = self.relu(self.linear1(x))
    x = self.relu(self.linear2(x))
    x = self.relu(self.linear3(x))
    x = self.relu(self.linear4(x))
    return self.linear5(x)

batch_size = 64
train_loader = DataLoader(datasets.MNIST('./datasets/',train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST('./datasets/',train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

model = NeuralNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(10):

  for i, (x,y) in enumerate(train_loader):
    x,y = Variable(x), Variable(y)

    optimizer.zero_grad()

    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    if i%10==0:
      print("Epoch: ", epoch, i, loss.item())

#After Training

with torch.no_grad():
  x,y = next(iter(test_loader))
  x,y = Variable(x).float(), Variable(y).long()
  output = model(x)
  pred = torch.max(output.data, 1)[1]
  accuracy = (y==pred).float().sum()/len(y)
  print("Accuracy on test data: ", accuracy.item()*100)
