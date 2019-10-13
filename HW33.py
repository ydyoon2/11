import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import os.path
import argparse

from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


parser = argparse.ArgumentParser()

# directory
parser.add_argument('--dataroot', type=str,
                    default="../data", help='path to dataset')
parser.add_argument('--ckptroot', type=str,
                    default="../checkpoint/ckpt.t7", help='path to checkpoint')

# hyperparameters settings
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--batch_size_train', type=int,
                    default=128, help='training set input batch size')
parser.add_argument('--batch_size_test', type=int,
                    default=64, help='test set input batch size')

# training settings
parser.add_argument('--resume', type=bool, default=False,
                    help='whether training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=True,
                    help='whether training using GPU')

# parse the arguments
opt = parser.parse_args()


# Data augmentation

print("==> Data Augmentation ...")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Loading CIFAR10

print("==> Preparing CIFAR10 dataset ...")

trainset = torchvision.datasets.CIFAR10(
    root=opt.dataroot, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=opt.batch_size_train, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=opt.dataroot, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=opt.batch_size_test, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Initialize CNN model

print("==> Initialize CNN model ...")

start_epoch = 0

# resume training from the last time
if opt.resume:
    # Load checkpoint
    print('==> Resuming from checkpoint ...')
    assert os.path.isdir(
        '../checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(opt.ckptroot)
    net = checkpoint['net']
    start_epoch = checkpoint['epoch']
else:
    # start over
    print('==> Building new CNN model ...')
    net = CNN()


# For training on GPU, we need to transfer net and data onto the GPU
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
if opt.is_gpu:
    net = net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.wd)


def calculate_accuracy(loader, is_gpu):
    """Calculate accuracy.

    Args:
        loader (torch.utils.data.DataLoader): training / test set loader
        is_gpu (bool): whether to run on GPU
    Returns:
        tuple: (overall accuracy, class level accuracy)
    """
    correct = 0.
    total = 0.

    for data in loader:
        images, labels = data
        if is_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total


print("==> Start training ...")

for epoch in range(start_epoch, opt.epochs + start_epoch):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        if opt.is_gpu:
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

        if epoch > 16:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]

    # Normalizing the loss by the total number of train batches
    running_loss /= len(trainloader)

    # Calculate training/test set accuracy of the existing model
    train_accuracy = calculate_accuracy(trainloader, opt.is_gpu)
    test_accuracy = calculate_accuracy(testloader, opt.is_gpu)

    print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(
        epoch+1, running_loss, train_accuracy, test_accuracy))

    # save model
    if epoch % 50 == 0:
        print('==> Saving model ...')
        state = {
            'net': net.module if opt.is_gpu else net,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '../checkpoint/ckpt.t7')

print('==> Finished Training ...')
