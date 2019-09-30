"""
HW3: Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. 
The convolution network should use (A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation. 
For full credit, the model should achieve 80-90% Test Accuracy. Submit via Compass (1) the code and 
(2) a paragraph (in a PDF document) which reports the results and briefly describes the model architecture. 
"""

import torch 
import torchvision 
import torch.utils.data 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable

#data augmentation
transform_train = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), 
                                      transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

#dropout
class CNN(nn.Module):
    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
                nn.Dropout2d(p=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.Dropout2d(p=0.1),
                nn.ReLU(inplace=True)
                )

        self.fc_layer = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 10)
                )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
    
net = CNN().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0005)

def calculate_accuracy(loader):
    correct = 0
    total = 0

    for data in loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total

for epoch in range(20):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if epoch > 3:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000
        optimizer.step()

        # print statistics
        running_loss += loss.data

    # Normalizing the loss by the total number of train batches
    running_loss /= len(trainloader)

    # Calculate training/test set accuracy of the existing model
    train_accuracy = calculate_accuracy(trainloader)
    test_accuracy = calculate_accuracy(testloader)

    print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(epoch+1, running_loss, train_accuracy, test_accuracy))
