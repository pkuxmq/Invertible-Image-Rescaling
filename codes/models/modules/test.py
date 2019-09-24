from Inv_arch import *
import numpy as np
import torch

xx = np.ndarray([1, 3, 4, 4])
xx = torch.tensor(xx)

for i in range(3):
    for j in range(4):
        for k in range(4):
            xx[0][i][j][k] = i * 16 + j * 4 + k

haar = HaarDownsampling(3)
shuffle = ShuffleChannel(12, 3)

y = haar.forward(xx)
yy = shuffle.forward(y)

yyy = shuffle.forward(yy, rev=True)

xxx = haar.forward(yyy, rev=True)

print(xx)
print(y)
print(yy)
print(yyy)
print(xxx)
