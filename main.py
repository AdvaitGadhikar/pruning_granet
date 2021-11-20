import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import numpy as np
from sparsify import *
from models import *

def train(model, optimizer, train_loader, epoch, mask=None, criterion, prune_iter):
    model.train()

    if epoch % prune_iter == 0:
        prune_ratio = get_pruning_ratio(args.num_epochs, epoch, args.si, args.sf, prune_iter)
        model, mask = global_prune(model, prune_ratio)

    epoch_loss = 0
    cnt = 0
    for i, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        apply_mask(model, mask)
        cnt += 1
        epoch_loss += loss

        if i % 10 == 0:
            preds = torch.argmax(output, dim=1)
            train_acc = (preds == labels).sum() / output.shape[0]
            print('Epoch: %.4f, Train Loss: %.4f, Train Accuracy: %.4f' %(epoch, loss, train_acc))

    epoch_loss = epoch_loss / cnt
    
    return model, mask

def evaluate(model, val_loader, mask=None, criterion):
    model.eval()
    val_acc = 0
    val_loss = 0
    cnt = 0
    for i, (data, labels) in enumerate(val_loader):
        output = model(data)
        loss = criterion(output, labels)
        preds = torch.argmax(output, dim=1)
        acc = (preds == labels).sum() / labels.shape[0]
        val_acc += acc
        cnt += 1
        val_loss += loss
    
    val_loss = val_loss / cnt
    val_acc = val_acc / cnt

    return val_loss, val_acc





def main():
    parser = argparse.ArgumentParser(description='PyTorch GraNet implementation')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=200, metavar='N',
                        help='number of epochs')
    parser.add_argument('--prune_every', type=int, default=10, metavar='N',
                        help='Prune every 10 epochs')
    parser.add_argument('--si', type=int, default=0.1, metavar='N',
                        help='initial sparsity')
    parser.add_argument('--sf', type=int, default=0.9, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--growth_ratio', type=float, default=0.1, metavar='M',
                        help='growth ratio')

    model = fc_model(784, 256, 10)
    
    train_data = torchvision.datasets.MNIST('data/', download = True, train = True, transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=args.batch_size,
                                          shuffle=True)

    val_data = torchvision.datasets.MNIST('data/', download = True, train = False, transform=torchvision.transforms.ToTensor())
    val_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=args.batch_size,
                                          shuffle=True)

    criterion = nn.CrossEntropyLoss()

    num_epochs = args.num_epochs
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)

    for epoch in range(num_epochs):
        
        model, mask = train(model, optimizer, train_loader, epoch, criterion, args.prune_every)      
        val_loss, val_acc = evaluate(model, val_loader, mask, criterion)
        print('Epoch: %.4f, Val Loss: %.4f, Val Accuracy: %.4f' %(epoch, val_loss, val_acc))


    

