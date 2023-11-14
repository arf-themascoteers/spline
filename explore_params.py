import torch
import torch.optim as optim
from ann import ANN
px = []
ann = ANN()
total_params = sum(p.numel() for p in ann.parameters())
print(total_params)#146

mp = sum(p.numel() for p in ann.machines.parameters())
print(mp)#5

lp = sum(p.numel() for p in ann.linear1.parameters())
print(lp)#141

print(mp+lp)#146
# param_group1 = {'params': px, 'lr': "0.01"}
# optimizer = optim.SGD([param_group1], lr=0.001)
