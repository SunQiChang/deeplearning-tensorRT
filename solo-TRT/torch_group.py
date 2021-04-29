import torch
input = torch.randn(size=(1,12,5,5))
m1=torch.nn.GroupNorm(3,12)
m2=torch.nn.GroupNorm(1,12)
m3=torch.nn.GroupNorm(3,12)
m1_out = m1(input)
m2_out = m2(input)

print('input:{}\nm1_out:{}\nm2_out:{}\n m1:{} m2:{} m3:{}\n'.
    format(input.shape, m1_out.shape, m2_out.shape,m1,m2, m3.weight ))

