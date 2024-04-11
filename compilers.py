# Python libs
import tvm
import torch
import torchvision
import numpy as np
from tvm import relay


if __name__ == "__main__":
    weights = np.load("assets/model.npz")