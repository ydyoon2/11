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
    
    
    
