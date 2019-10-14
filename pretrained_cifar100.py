import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# Loading the data

def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='~/scratch/'))
        model = FineTune(model, num_classes=100)
    return model


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
        classes: classes names
    """
    # Data Augmentation
    print("==> Data Augmentation ...")

    # Normalize training set together with augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


def train_model(net, optimizer, scheduler, criterion, trainloader,
                testloader, start_epoch, epochs, is_gpu):
    """
    Training process.

    Args:
        net: ResNet model
        optimizer: Adam optimizer
        criterion: CrossEntropyLoss
        trainloader: training set loader
        testloader: test set loader

        epochs: training epochs
        is_gpu: whether use GPU

    """
    print("==> Start training ...")

    # switch to train mode
    net.train()

    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        scheduler.step()

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if is_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
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

    print('==> Finished Training ...')






class FineTune(nn.Module):
    """Fine-tune pre-trained ResNet model."""

    def __init__(self, resnet, num_classes):
        """Initialize Fine-tune ResNet model."""
        super(FineTune, self).__init__()

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )

        # # Freeze those weights
        # for param in self.features.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        """Forward pass of fint-tuned of ResNet-18 model."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out





parser = argparse.ArgumentParser()

# directory
parser.add_argument('--dataroot', type=str,
                    default="../data", help='path to dataset')


# hyperparameters settings
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float,
                    default=1e-5, help='weight decay (L2 penalty)')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size_train', type=int,
                    default=64, help='training set input batch size')
parser.add_argument('--batch_size_test', type=int,
                    default=64, help='test set input batch size')

# training settings
parser.add_argument('--resume', type=bool, default=True,
                    help='whether re-training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=True,
                    help='whether training using GPU')

# model_urls
parser.add_argument('--model_url', type=str, default="https://download.pytorch.org/models/resnet18-5c106cde.pth",
                    help='model url for pretrained model')

# parse the arguments
args = parser.parse_args()


def main():
    """Main pipeline for training ResNet model on CIFAR100 Dataset."""
    start_epoch = 0


        # start over
    print('==> Load pre-trained ResNet model ...')
    net = resnet18(args.model_url)

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if args.is_gpu:
        net = net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Loss function, optimizer for fine-tune-able params
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # data loader for CIFAR100
    trainloader, testloader = data_loader(args.dataroot,
                                          args.batch_size_train,
                                          args.batch_size_test)

    # train pre-trained model on CIFAR100
    train_model(net,
                optimizer,
                scheduler,
                criterion,
                trainloader,
                testloader,
                start_epoch,
                args.epochs,
                args.is_gpu)


if __name__ == '__main__':
    main()
