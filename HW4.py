"""
Deep convolutional networks using ResNets
"""

import torch 
import torchvision 
import torch.nn as nn
import torch.utils.data 
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms


batch_size = 32
num_epochs = 10

transform_train = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), 
                                      transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))])

# For trainning data
trainset = torchvision.datasets.CIFAR100(root='~/scratch/', train=True,download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

# For testing data
testset = torchvision.datasets.CIFAR100(root='~/scratch/', train=False,download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
