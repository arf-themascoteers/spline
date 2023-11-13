import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


ar = torch.linspace(-1,1,100)
print(F.sigmoid(ar))

plt.plot(ar, F.sigmoid(ar))
plt.show()