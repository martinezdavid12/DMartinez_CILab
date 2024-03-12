'''
store frame every k, k+n, where  1<n<len(video)
for every frame not stored/represented, interpolate frame k and k+n to initialize model optimization
optionally, use MSE to determing fraction/proportion of difference to use. 
i.e. if frame k+2 has very little difference with frame k, apply the following interpolation:

interpolate(k+x) = params(k) + (MSE(k, k+x)/MSE(k, k+n)) * (params(k+n) - params(k))

for MSE calculations with any previous frame, use prefix sum. (currently using k)
steps:
1. collect prefix sum of MSE
2. perform naive interpolation without weights based on MSE
3. Compare to MSE-based heurstic.
4. Generate special chart comparing MSE to time to reach certain PSNRs.
5. Can we learn optimal
'''
import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer

n = 10 # number of frames to skip on initial fittings

def interpolate_unweighted(model_a, model_b):
    # for every param(except hash table), find linear interpolation of model_a and model_b)
    # perform model_a+model_b/2 -> mean value?
    interpolated_model = hashNerf(32, 64, 3)
    params_new = list(interpolated_model.parameters())
    params_1 = list(model_b.parameters())
    params_0 = list(model_a.parameters())   
    with torch.no_grad():
        i = 0
        for param in interpolated_model.parameters():
            param.copy_(torch.tensor((params_0[i]+params_1[i])/2.0).type(torch.float32))
            i += 1
    return interpolated_model

def heuristic_interpolate(model_a, model_b, MSE_w):
    interpolated_model = hashNerf(32, 64, 3)
    params_new = list(interpolated_model.parameters())
    params_1 = list(model_b.parameters())
    params_0 = list(model_a.parameters())   
    with torch.no_grad():
        i = 0
        for param in interpolated_model.parameters():
            new_weight = torch.tensor(params_0 + (MSE_w)*(params_0[i]+params_1[i])/2.0).type(torch.float32)
            param.copy_()
            i += 1
    return interpolated_model


