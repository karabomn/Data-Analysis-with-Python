import numpy as np
import torch
from torch import nn

def create_model():
    # your code here
    NN = nn.Sequential(nn.Linear(784, 256, bias = True),
                       nn.ReLU(),
                       nn.Linear(256, 16, bias = True),
                       nn.ReLU(),
                       nn.Linear(16, 10, bias = True))
    return NN

def count_parameters(model):
    # your code here
    i = 0
    for param in model.parameters():
        i += np.count_nonzero(param.detach().numpy())
    return i
