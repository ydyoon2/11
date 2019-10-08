import os
import torch
import torchvision
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import argparse

#from resnet import resnet_cifar



import torch.utils.data


from torch.autograd import Variable



def initialize_weights(module):
    """Initialize weights."""
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal(module.weight.data)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 kernel size with padding convolutional layer in ResNet BasicBlock."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def resnet_cifar(**kwargs):
    """Initialize ResNet model."""
    model = ResNet(BasicBlock, [2, 4, 4, 2], **kwargs)
    return model


class BasicBlock(nn.Module):
    """Basic Block of ReseNet."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """Basic Block of ReseNet Builder."""
        super(BasicBlock, self).__init__()

        # First conv3x3 layer
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        #  Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        # ReLU Activation Function
        self.relu = nn.ReLU(inplace=True)

        # Second conv3x3 layer
        self.conv2 = conv3x3(out_channels, out_channels)

        #  Batch Normalization
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        # downsample for `residual`
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward Pass of Basic Block."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class ResNet(nn.Module):
    """Residual Neural Network."""

    def __init__(self, block, duplicates, num_classes=100):
        """Residual Neural Network Builder."""
        super(ResNet, self).__init__()

        self.in_channels = 32
        self.conv1 = conv3x3(in_channels=3, out_channels=32)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)

        # block of Basic Blocks
        self.conv2_x = self._make_block(block, duplicates[0], out_channels=32, stride=1, padding=1)
        self.conv3_x = self._make_block(block, duplicates[1], out_channels=64, stride=2, padding=1)
        self.conv4_x = self._make_block(block, duplicates[2], out_channels=128, stride=2, padding=1)
        self.conv5_x = self._make_block(block, duplicates[3], out_channels=256, stride=2, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc_layer = nn.Linear(256, num_classes)

        # initialize weights
        # self.apply(initialize_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_block(self, block, duplicates, out_channels, stride=1, padding = 1):
        """
        Create Block in ResNet.

        Args:
            block: BasicBlock
            duplicates: number of BasicBlock
            out_channels: out channels of the block

        Returns:
            nn.Sequential(*layers)
        """
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
        """Forward pass of ResNet."""
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Stacked Basic Blocks
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)

        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)

        return out



















def data_loader(dataroot, batch_size_train, batch_size_test):
    """
    Data Loader for CIFAR100 Dataset.

    Args:
        dataroot: data root directory
        batch_size_train: mini-Batch size of training set
        batch_size_test: mini-Batch size of test set

    Returns:
        trainloader: training set loader
        testloader: test set loader
    """
    # Data Augmentation
    print("==> Data Augmentation ...")

    # Normalize training set together with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    # Loading CIFAR100
    print("==> Preparing CIFAR100 dataset ...")

    trainset = torchvision.datasets.CIFAR100(root=dataroot,
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root=dataroot,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=4)

    return trainloader, testloader


def calculate_accuracy(net, loader, is_gpu):
    """
    Calculate accuracy.

    Args:
        loader (torch.utils.data.DataLoader): training / test set loader
        is_gpu (bool): whether to run on GPU

    Returns:
        tuple: overall accuracy
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


def train(net, criterion, optimizer, trainloader,
          testloader, start_epoch, epochs, is_gpu):
    """
    Training process.

    Args:
        net: ResNet model
        criterion: CrossEntropyLoss
        optimizer: SGD with momentum optimizer
        trainloader: training set loader
        testloader: test set loader
        start_epoch: checkpoint saved epoch
        epochs: training epochs
        is_gpu: whether use GPU

    """
    print("==> Start training ...")

    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data

            if is_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        # Calculate training/test set accuracy of the existing model
        train_accuracy = calculate_accuracy(net, trainloader, is_gpu)
        test_accuracy = calculate_accuracy(net, testloader, is_gpu)

        print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(
            epoch+1, running_loss, train_accuracy, test_accuracy))

        # save model
        if epoch % 50 == 0:
            print('==> Saving model ...')
            state = {
                'net': net.module if is_gpu else net,
                'epoch': epoch,
            }
            if not os.path.isdir('../checkpoint'):
                os.mkdir('../checkpoint')
            torch.save(state, '../checkpoint/ckpt.t7')

    print('==> Finished Training ...')
































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
            '../checkpoint'), 'Error: no checkpoint directory found!'
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
