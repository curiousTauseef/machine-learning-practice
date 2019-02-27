import numpy as np

class LinearRegressionModel():
  '''Liner Regression: y = wx without any bias'''

  def __init__(self, learning_rate, epochs):
    self._learning_rate = learning_rate
    self._w = np.random.randint(0,10)
    self._epochs = epochs

  def forward(self, x):
    return self._w*x
  
  def loss(self, x, y):
    return (self.forward(x)-y)**2
  
  def gradient(self, x, y):
    return (2*x)*((self._w*x) - y)

  def fit(self, x_data, y_data):
    for epoch in range(self._epochs):
      loss_val = []
      for x,y in zip(x_data, y_data):
        self._w = self._w - (self._learning_rate*self.gradient(x, y))
        loss_val.append(self.loss(x,y))

      #print("epoch, mean squared error: ", epoch, np.average(loss_val))
  
  def predict(self, x):
    return self.forward(x)


x_data = [1.0,2.0,3.0,4.0]
y_data = [2.0,4.0,6.0,8.0]

model = LinearRegressionModel(0.01, 100)
model.fit(x_data, y_data)
print("After Training Prediction (5):", model.predict(5))






