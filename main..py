import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='PyTorch GraNet implementation')
    