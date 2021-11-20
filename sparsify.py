import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


def global_prune(model, pruning_ratio):
    # use torch.count_nonzero to get the number of non zero elements in a tensor
    nonzero_params = 0
    weight_magnitude = []

    for name, weight in model.named_parameters():
        nonzero_params += torch.count_nonzero(weight.data)  
        weight_magnitude.append(torch.abs(weight))   
    num_retain = int((1 - pruning_ratio) * nonzero_params)
    weight_vals = torch.cat([torch.flatten(x) for x in weight_magnitude])
    values, indices = torch.sort(weight_vals, descending = True)
    threshold = values[num_retain]

    mask_list = []
    for name, weight in model.named_parameters():
        mask = torch.zeros_like(weight)
        mask = torch.where(torch.abs(weight) >= threshold, 1, 0)
        weight.data = weight.data * mask
        mask_list.append(mask)
    
    growth(model, growth_ratio)

    return model, mask_list

def get_pruning_ratio(num_epochs, curr_epoch, si, sf, prune_every):
    t0 = 0
    prune_ratio = sf + (si - sf)*(1 - (curr_epoch / (num_epochs)))**3
    return prune_ratio

def growth(model, growth_ratio):
    """
    growth_ratio is the proportion of damaged weights that need to be replaced
    prunes by magnitude and grows by magnitude of gradient
    """
    mask_list = []
    grad_list = []
    for name, weight in model.named_parameters():
        nonzero_elems += torch.count_nonzero(weight.data)
        num_retain = int((1-growth_ratio)*nonzero_elems)
        vals, idx = torch.sort(torch.abs(weight), descending=True)
        threshold = vals[num_retain]
        mask = torch.where(torch.abs(weight) >= threshold, 1, 0)
        mask_list.append(mask)
        grad_list.append(weight.grad.data.clone())
    
    step = 0
    for name, weight in model.named_parameters():
        nz = torch.count_nonzero(weight.data)
        grad = grad_list[step]
        mask = (mask_list[step] == 0)
        grad_val, idx = torch.sort(torch.abs(grad * mask), descending = True)
        num_regrow = nz - torch.count_nonzero(mask_list[step])
        mask_list[step][idx[:num_regrow]] = 1
    
    apply_mask(model, mask_list)

    return model

def apply_mask(model, mask_list):
    if mask_list == None:
        mask_list = []
        for name, weight in model.named_parameters():
            mask_list.append(torch.ones_like(weight))
            
    step = 0
    for name, weight in model.named_parameters():
        weight.data = weight.data * mask_list[step]
        step += 1
        