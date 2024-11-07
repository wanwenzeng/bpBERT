import numpy as np
import torch

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-2, keepdims=True))
    return e_x / e_x.sum(axis=-2, keepdims=True)

def sigmoid(x):
    m = torch.nn.Sigmoid()
    return m(x)

def mean(x):
    return sum(x) / len(x)

def clipped_exp(x, min_value=-50, max_value=50):
    return torch.exp(torch.clamp(x, min_value, max_value))
