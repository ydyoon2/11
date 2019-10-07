"""
Deep convolutional networks using ResNets
"""

import torch 
import torchvision 
import torch.nn as nn
import torch.utils.data 
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import os

# Load CIFAR100 dataset
batch_size = 32
num_epochs = 10

transform_train = transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))])
# For trainning data
trainset = torchvision.datasets.CIFAR100(root='~/scratch/', train=True,download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
# For testing data
testset = torchvision.datasets.CIFAR100(root='~/scratch/', train=False,download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

# Access the data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        
# TinyImageNet
def create_val_folder(val_dir):
    """
    This method is responsible for separating validation
    images into separate sub folders
    """
    # path where validation data is present now
    path = os.path.join(val_dir, 'images')
    # file where image2class mapping is present
    filename = os.path.join(val_dir, 'val_annotations.txt')
    fp = open(filename, "r") # open file in read mode
    data = fp.readlines() # read line by line
    """
    Create a dictionary with image names as key and
    corresponding classes as values
    """
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath): # check if folder exists
            os.makedirs(newpath)
        # Check if image exists in default directory
        if os.path.exists(os.path.join(path, img)):
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return

train_dir = '/u/training/tra318/scratch/tiny-imagenet-200/train'
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
#print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_dir = '/u/training/tra318/scratch/tiny-imagenet-200/val/images'
if 'val_' in os.listdir(val_dir)[0]:
    create_val_folder(val_dir)
else:
    pass
val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
#print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for images, labels in train_loader:
    images = Variable(images).to(device)
    labels = Variable(labels).to(device)
for images, labels in val_loader:
    images = Variable(images).to(device)
    labels = Variable(labels).to(device)
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample = None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
            
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x) # 3*3, stride = 1
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out) # 3*3, stride = 1
        out = self.bn2(out)
        
        if self.downsample is not None: #downsample: when the shape of identity != shape of out
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, basic_block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        
        self.in_channels = 32
        self.conv1 = conv3x3(3, 32, 3, 1, 1) #input, output, kernel, stride, padding
        self.bn1 = nn.BatchNorm2d(32) #feature
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)
        
        self.layer1 = self._make_layer(basic_block, 64, num_blocks[0], stride=1, padding=1)
        self.layer2 = self._make_layer(basic_block, 128, num_blocks[1], stride=2, padding=1)
        self.layer3 = self._make_layer(basic_block, 256, num_blocks[2], stride=2, padding=1)
        self.layer4 = self._make_layer(basic_block, 512, num_blocks[3], stride=2, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc_layer = nn.Linear(256, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion: 
            
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks): 
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.maxpool(x)
        x = self.fc(x)
        
        return x
