
import torch

from torch import nn


x=torch.rand(size=(80,20))
x=x.reshape(1,1,x.shape[0],x.shape[1])
print("输出:",x.shape)
