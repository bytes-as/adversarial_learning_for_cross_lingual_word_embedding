import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
N, D_in, H, D_out = 100000, 2, 1, 2


X = []
y = []
for i in range(100):
	X.append([[i, 20*i]])
	y.append([[i + 20*i, 10*i]])

X = Variable(torch.Tensor(np.array(X)))
y = Variable(torch.Tensor(np.array(y)))
# X = Variable(torch.randn(N, D_in))
# y = Variable(torch.randn(N, D_out))

print(X.size())
print(y.size())

class NN(nn.Module):
	def __init__(self, D_in, H, D_out):
		super(NN, self).__init__()
		self.linear = nn.Linear(D_in, D_out, bias=False)
		print('linear: ', self.linear)
	def forward(self, X):
		# print(X)
		x = self.linear(X)
		# print(x)
		return x

model = NN(D_in, H, D_out)

# model = torch.nn.Sequential(
#     # nn.LeakyReLU(0, inplace=True)
#     torch.nn.Linear(D_in, D_out),
#     # torch.nn.ReLU(),
#     # torch.nn.Linear(H, D_out),
# )

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(100, 80, bias=False)
#         init.normal(self.fc1.weight, mean=0, std=1)
#         self.fc2 = nn.Linear(80, 87)
#         self.fc3 = nn.Linear(87, 94)
#         self.fc4 = nn.Linear(94, 100)

#     def forward(self, x):
#          x = self.fc1(x)
#          x = F.relu(self.fc2(x))
#          x = F.relu(self.fc3(x))
#          x = F.relu(self.fc4(x))
#          return x

loss_fn = torch.nn.MSELoss()

learning_rate = 1e-1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for p in model.parameters():
	a = p[0].data.numpy()
	print('  var['+str(0)+']: ', a, end ='')
for t in range(1):
	y_pred = model(X)
	loss = loss_fn(y_pred, y)
	l = loss
	if t%100 == 49 or  t%100==99:
		print('\nepoch: %-8d'%t, ' loss =%-15f'%l.data.numpy()[0], end=' ')
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	count = 0
	if t%100 == 49 or  t%100==99:
		# for p in model.parameters():
			# a = p[0].data.numpy()
			# print('  var[%-2d]: %-24s'%(count, a), end ='')
		w = model.linear.weight.data.numpy()
		print(w)
			# count += 1
	
	