import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim
import torch.backends.cudnn as cudnn
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

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.434, 0.495, 0.412], std=[0.259, 0.259, 0.243])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.434, 0.495, 0.412], std=[0.259, 0.259, 0.243])
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


def calculate_accuracy(net, loader):
    correct = 0.
    total = 0.

    for data in loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100 * correct / total

def train_model(net, optimizer, scheduler, criterion, trainloader, testloader, start_epoch, epochs):
    #net.train()
    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        scheduler.step()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        running_loss /= len(trainloader)
        train_accuracy = calculate_accuracy(net, trainloader)
        test_accuracy = calculate_accuracy(net, testloader)

        print('Epoch {}, Loss: {:.4f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}'
          .format(epoch, loss.item(), train_accuracy, test_accuracy))


class FineTune(nn.Module):
    def __init__(self, resnet, num_classes):
        super(FineTune, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


start_epoch = 0
net = resnet18("https://download.pytorch.org/models/resnet18-5c106cde.pth")
net = net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5, gamma=0.2)
trainloader, testloader = data_loader('~/scratch/', 64, 64)
train_model(net, optimizer, scheduler, criterion, trainloader, testloader, start_epoch, 10)

