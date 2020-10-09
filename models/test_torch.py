import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable


def normalize(normals):
    magnitude = torch.pow(normals, 2).sum(1)
    magnitude = magnitude.sqrt().repeat(1,3,1,1)
    normed = normals / (magnitude + 1e-6)
    return normed

