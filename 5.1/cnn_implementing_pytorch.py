import torch
import numpy as np

array = np.array([[1,2,3], [4,5,6]])
tensor = torch.Tensor(array)
print(tensor.shape)