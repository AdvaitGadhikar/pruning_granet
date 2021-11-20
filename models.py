import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class fc_model(nn.Module):
    def __init__(self, input_features, num_classes, num_hidden):
        super(fc_model, self).__init__()

        self.linear1 = nn.Linear(input_features, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x

