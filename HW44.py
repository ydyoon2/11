import os
import torch
import torchvision
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import argparse
import torch.utils.data
from torch.autograd import Variable

def data_loader(dataroot, batch_size_train, batch_size_test):    
    transform_train = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(), 
                                          transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))])
    transform_test = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))])
    trainset = torchvision.datasets.CIFAR100(root='~/scratch/', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR100(root='~/scratch/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=8)
    return trainloader, testloader

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal(module.weight.data)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def resnet_cifar(**kwargs):
    model = ResNet(BasicBlock, [2, 4, 4, 2], **kwargs)
    return model


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, basic_block, num_blocks, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 32
        self.conv1 = conv3x3(3, 32)
        self.bn1 = nn.BatchNorm2d(32) #feature
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)
        
        self.conv2_x = self._make_block(basic_block, 32, num_blocks[0], stride=1)
        self.conv3_x = self._make_block(basic_block, 64, num_blocks[1], stride=2)
        self.conv4_x = self._make_block(basic_block, 128, num_blocks[2], stride=2)
        self.conv5_x = self._make_block(basic_block, 256, num_blocks[3], stride=2)
        
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc_layer = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_block(self, basic_block, num_blocks, out_channels, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, duplicates):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)

        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)

        return out


def calculate_accuracy(net, loader, is_gpu):
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


def train(net, criterion, optimizer, trainloader,
          testloader, start_epoch, epochs, is_gpu):

    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            if is_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            inputs, labels = Variable(inputs), Variable(labels)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.data

        running_loss /= len(trainloader)

        train_accuracy = calculate_accuracy(net, trainloader, is_gpu)
        test_accuracy = calculate_accuracy(net, testloader, is_gpu)

        print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(
            epoch+1, running_loss, train_accuracy, test_accuracy))


parser = argparse.ArgumentParser()

# directory
parser.add_argument('--dataroot', type=str,
                    default="../data", help='path to dataset')
parser.add_argument('--ckptroot', type=str,
                    default="../checkpoint/ckpt.t7", help='path to checkpoint')

# hyperparameters settings
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float,
                    default=0.9, help='momentum factor')
parser.add_argument('--weight_decay', type=float,
                    default=1e-5, help='weight decay (L2 penalty)')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size_train', type=int,
                    default=256, help='training set input batch size')
parser.add_argument('--batch_size_test', type=int,
                    default=256, help='test set input batch size')

# training settings
parser.add_argument('--resume', type=bool, default=False,
                    help='whether re-training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=True,
                    help='whether training using GPU')

# parse the arguments
args = parser.parse_args()


def main():
    """Main pipeline for training ResNet model on CIFAR100 Dataset."""
    start_epoch = 0

    # resume training from the last time
    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint ...')
        assert os.path.isdir(
            '~/checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.ckptroot)
        net = checkpoint['net']
        start_epoch = checkpoint['epoch']
    else:
        # start over
        print('==> Building new ResNet model ...')
        net = resnet_cifar()

    print("==> Initialize CUDA support for ResNet model ...")

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if args.is_gpu:
        # net = net.cuda()
        # net = torch.nn.DataParallel(
        #     net, device_ids=range(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    # Loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Load CIFAR100
    trainloader, testloader = data_loader(args.dataroot,
                                          args.batch_size_train,
                                          args.batch_size_test)

    # training
    train(net,
          criterion,
          optimizer,
          trainloader,
          testloader,
          start_epoch,
          args.epochs,
          args.is_gpu)


if __name__ == '__main__':
    main()
