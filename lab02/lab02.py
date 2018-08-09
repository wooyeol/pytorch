import torch
from torch.autograd import Variable
from torch import nn

torch.manual_seed(3)

x_train = torch.Tensor([[1],[2],[3]])
y_train = torch.Tensor([[1],[2],[3]])

x, y = Variable(x_train), Variable(y_train)

W = Variable(torch.rand(1,1))
x.mm(W)

cost_func = nn.MSELoss()

lr = 0.01

for step in range(300):
    prediction = x.mm(W)
    cost = cost_func(prediction, y)
    gradient = (prediction-y).view(-1).dot(x.view(-1)) / len(x)
    W -= lr * gradient
    
    if step %10 == 0:
        print(step, "going cost")
        print(cost)
        print((prediction-y).view(-1))
        print((x.view(-1)))
        print(gradient)
        print(W)