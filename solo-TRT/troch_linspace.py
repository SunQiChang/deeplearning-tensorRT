import torch

x_range=torch.linspace(-1,1,20)
y_range=torch.linspace(-1,1,20)
y,x=torch.meshgrid(y_range, x_range)
print(x)
print('=============================')
print(y)
