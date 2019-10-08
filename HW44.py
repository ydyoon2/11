class ResNet(nn.Module):
    def __init__(self, basic_block, num_blocks, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 32
        self.conv1 = conv3x3(3, 32, 3) #input, output, kernel, stride, padding
        self.bn1 = nn.BatchNorm2d(32) #feature
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)
        
        self.layer1 = self._make_layer(basic_block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(basic_block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(basic_block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(basic_block, 512, num_blocks[3], stride=2)
        
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc_layer = nn.Linear(256, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    
    def _make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None
        
        if stride != 1 or self.in_channels != planes * block.expansion: 
            
            downsample = nn.Sequential(
                conv3x3(self.in_channels, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        
        self.in_channels = planes * block.expansion
        
        for _ in range(1, blocks): 
            layers.append(block(self.in_channels, planes))

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
