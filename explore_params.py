import torch
import torch.optim as optim
from ann import ANN
px = []
ann = ANN()
total_params = sum(p.numel() for p in ann.parameters())
print(total_params)#146
# for params in ann.machines:
#     for p in params.parameters():
#         px.append(p)
#
# param_group1 = {'params': px, 'lr': "0.01"}
# optimizer = optim.SGD([param_group1], lr=0.001)
