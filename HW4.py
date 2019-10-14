import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch.nn as nn
from torch.autograd import Variable

num_epochs = 1
batch_size = 128

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
])

def create_val_folder(val_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(val_dir, 'images')  # path where validation data is present now
    filename = os.path.join(val_dir, 'val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)
        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return

# Your own directory to the train folder of tiyimagenet
train_dir = '/u/training/tra318/scratch/tiny-imagenet-200/train/'
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
# To check the index for each classes
# print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
# Your own directory to the validation folder of tiyimagenet
val_dir = '/u/training/tra318/scratch/tiny-imagenet-200/val/'


if 'val_' in os.listdir(val_dir+'images/')[0]:
    create_val_folder(val_dir)
    val_dir = val_dir+'images/'
else:
    val_dir = val_dir+'images/'


val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
# To check the index for each classes
# print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


# YOUR CODE GOES HERE
# Change to your ResNet 

# BasicBlock
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

# ResNet
class ResNet(nn.Module):
    def __init__(self, basic_block, num_blocks, num_classes=200):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = conv3x3(in_channels=3, out_channels=64)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

        self.conv2_x = self._make_block(basic_block, num_blocks[0], out_channels=64, stride=1, padding=1)
        self.conv3_x = self._make_block(basic_block, num_blocks[1], out_channels=128, stride=2, padding=1)
        self.conv4_x = self._make_block(basic_block, num_blocks[2], out_channels=256, stride=2, padding=1)
        self.conv5_x = self._make_block(basic_block, num_blocks[3], out_channels=512, stride=2, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=5, stride=1)
        self.fc_layer = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_block(self, basic_block, num_blocks, out_channels, stride=1, padding = 1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )

        layers = []
        layers.append(
            basic_block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(basic_block(out_channels, out_channels))

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
        out = self.maxpool_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)

        return out

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def accuracy(net, loader):
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








resnet = ResNet(BasicBlock, [2, 4, 4, 2], 200)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(num_epochs):
    # Train the model
    resnet.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_loader):
        X_train_batch,Y_train_batch = X_train_batch.to(device),Y_train_batch.to(device)
       
       
    # Test the model
    with torch.no_grad():
        resnet.eval()
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(val_loader):
            X_test_batch, Y_test_batch= X_test_batch.to(device),Y_test_batch.to(device)

    train_accuracy = accuracy(resnet, train_loader)
    test_accuracy = accuracy(resnet, val_loader)
    print("epoch: {}, train_accuracy: {}%, test_accuracy: {}%".format(epoch, train_accuracy, test_accuracy))


