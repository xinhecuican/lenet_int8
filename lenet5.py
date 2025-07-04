# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:28:48 2024

@author: Phat_Dang
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
#import adamod
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1) # [28, 28]
        self.relu1 = nn.ReLU() # [6, 28, 28]
        self.conv2 = nn.Conv2d(6, 16, 3, 1) # [6, 14, 14]
        self.relu2 = nn.ReLU() # [16, 12, 12]
        self.pooling = nn.AvgPool2d(2,2) # [16, 6, 6]
        self.fc1 = nn.Linear(576, 10)
        # self.relu3 = nn.ReLU()
        # self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.relu3(x)
        # x = self.fc3(x)
        output = torch.log_softmax(x, dim=1)
        return output

def save_weights_to_txt(model):
    weights_dir = "data/weights/"  # Directory to save weights
    import os
    os.makedirs(weights_dir, exist_ok=True)  # Create directory if not exists

    # Save convolutional layer 1 weights and bias
    np.savetxt(weights_dir + "w_conv1.txt", model.state_dict()['conv1.weight'].cpu().numpy().reshape(-1), fmt='%f')
    np.savetxt(weights_dir + "b_conv1.txt", model.state_dict()['conv1.bias'].cpu().numpy(), fmt='%f')

    # Save convolutional layer 2 weights and bias
    np.savetxt(weights_dir + "w_conv2.txt", model.state_dict()['conv2.weight'].cpu().numpy().reshape(-1), fmt='%f')
    np.savetxt(weights_dir + "b_conv2.txt", model.state_dict()['conv2.bias'].cpu().numpy(), fmt='%f')

    # Save fully connected layer 1 weights and bias
    np.savetxt(weights_dir + "w_fc1.txt", model.state_dict()['fc1.weight'].cpu().numpy().reshape(-1), fmt='%f')
    np.savetxt(weights_dir + "b_fc1.txt", model.state_dict()['fc1.bias'].cpu().numpy(), fmt='%f')

    # Save fully connected layer 2 weights and bias
    # np.savetxt(weights_dir + "w_fc2.txt", model.state_dict()['fc2.weight'].cpu().numpy().reshape(-1), fmt='%f')
    # np.savetxt(weights_dir + "b_fc2.txt", model.state_dict()['fc2.bias'].cpu().numpy(), fmt='%f')

    # Save fully connected layer 3 weights and bias
    # np.savetxt(weights_dir + "w_fc3.txt", model.state_dict()['fc3.weight'].cpu().numpy().reshape(-1), fmt='%f')
    # np.savetxt(weights_dir + "b_fc3.txt", model.state_dict()['fc3.bias'].cpu().numpy(), fmt='%f')

    print("Weights and biases saved to", weights_dir)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = LeNet5().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    ### Training Phase
    start_time = time.perf_counter()
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    # Code training của bạn ở đây
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Training time: {execution_time:.2f} seconds")

    ### Save the model
    if args.save_model:
        torch.save(model.state_dict(), "data/lenet5_model.pt")
    
    save_weights_to_txt(model)
    
    ### Load the model from lenet5_model.pt file
    print(model)
    model.load_state_dict(torch.load("data/lenet5_model.pt"))
    keys = model.state_dict().keys()
    
    
    ### Export all model's weights to *.txt file
    
    for steps, weights in enumerate(keys):
        w = model.state_dict()[weights].data
        w_reshaped = w.reshape(w.shape[0],-1)
        np.savetxt(r'data/weights/' + str(weights) + '.txt', w_reshaped.cpu(), fmt='%f')
     
    ### Inference Phase
    print('Testing...')
    test(model, device, test_loader)
    
if __name__ == '__main__':
   main()



