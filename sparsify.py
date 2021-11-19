import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


def global_prune(model, pruning_ratio):
    # use torch.count_nonzero to get the number of non zero elements in a tensor
    nonzero_params = 0
    weight_magnitude = []
    for name, weight in module.named_parameters():
        nonzero_params += torch.count_nonzero()  
        weight_magnitude.append      
    num_retain = int((1 - pruning_ratio) * nonzero_params)



return pruned_model, model_mask