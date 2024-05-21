'''
Pytorch implementation of ResNet models.

Reference:
[1] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR, 2016.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet110', 'resnet152', 'lenet', 'densenet121', 'densenet169', 'densenet201', 'densenet201','resnet18_distillation', 'Linear','vgg11', 'MLP'
]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        return out

        
    def forward1_(self, x):
        hiddens = []
        out = F.relu(self.bn1(self.conv1(x)))
        hiddens.append(out)
        out = self.layer1[0](out)
        hiddens.append(out)
        out = self.layer1[1](out)
        hiddens.append(out)
        out = self.layer1[2](out)
        hiddens.append(out)
        out = self.layer2[0](out)
        hiddens.append(out)
        out = self.layer2[1](out)
        hiddens.append(out)
        out = self.layer2[2](out)
        hiddens.append(out)
        out = self.layer2[3](out)
        hiddens.append(out)
        out = self.layer3[0](out)
        hiddens.append(out)
        out = self.layer3[1](out)
        hiddens.append(out)
        out = self.layer3[2](out)
        hiddens.append(out)
        out = self.layer3[3](out)
        hiddens.append(out)
        out = self.layer3[4](out)
        hiddens.append(out)
        out = self.layer3[5](out)
        hiddens.append(out)
        out = self.layer4[0](out)
        hiddens.append(out)
        out = self.layer4[1](out)
        hiddens.append(out)
        out = self.layer4[2](out)
        hiddens.append(out)
        return hiddens
        
    def forward1__(self, x):
        hiddens = []
        out = F.relu(self.bn1(self.conv1(x)))
        hiddens.append(out)
        out = self.layer1[0](out)
        hiddens.append(out)
        out = self.layer1[1](out)
        hiddens.append(out)
        out = self.layer1[2](out)
        hiddens.append(out)
        out = self.layer2[0](out)
        hiddens.append(out)
        out = self.layer2[1](out)
        hiddens.append(out)
        out = self.layer2[2](out)
        hiddens.append(out)
        out = self.layer2[3](out)
        hiddens.append(out)
        out = self.layer3[0](out)
        hiddens.append(out)
        out = self.layer3[1](out)
        hiddens.append(out)
        out = self.layer3[2](out)
        hiddens.append(out)
        out = self.layer3[3](out)
        hiddens.append(out)
        out = self.layer3[4](out)
        hiddens.append(out)
        out = self.layer3[5](out)
        hiddens.append(out)
        out = self.layer4[0](out)
        hiddens.append(out)
        out = self.layer4[1](out)
        hiddens.append(out)
        out = self.layer4[2](out)
        out = F.avg_pool2d(out, 4)
        hiddens.append(out)
        return hiddens


    def forward1(self, x):
        hiddens = []
        out = F.relu(self.bn1(self.conv1(x)))
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer1[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer1[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer4[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer4[1](out)
        hiddens.append(out.view(out.size(0), -1))
        return hiddens

    def forward1(self, x):
        hiddens = []
        out = F.relu(self.bn1(self.conv1(x)))
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer1[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer1[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer1[2](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[2](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[3](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[2](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[3](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[4](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[5](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer4[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer4[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer4[2](out)
        hiddens.append(out.view(out.size(0), -1))
        return hiddens
        

    def forward1(self, x):
        hiddens = []
        out = F.relu(self.bn1(self.conv1(x)))
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer1[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer1[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer1[2](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[2](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer2[3](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[2](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[3](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[4](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[5](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[6](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[7](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[8](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[9](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[10](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[11](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[12](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[13](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[14](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[15](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[16](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[17](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[18](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[19](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[20](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[21](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[22](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[23](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[24](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer3[25](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer4[0](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer4[1](out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.layer4[2](out)
        hiddens.append(out.view(out.size(0), -1))
        return hiddens
        
    def forward1_(self, x):
        hiddens = []
        out = F.relu(self.bn1(self.conv1(x)))
        hiddens.append(out)# new add
        out = self.layer1[0](out)
        hiddens.append(out)# new add
        out = self.layer1[1](out)
        hiddens.append(out)# new add
        out = self.layer2[0](out)
        hiddens.append(out)
        out = self.layer2[1](out)
        hiddens.append(out)
        out = self.layer3[0](out)
        hiddens.append(out)
        out = self.layer3[1](out)
        hiddens.append(out)
        out = self.layer4[0](out)
        hiddens.append(out)
        out = self.layer4[1](out)
        hiddens.append(out)
        return hiddens
        
    '''
    def feature1(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out)
        out = self.layer1[1](out)
        out = self.layer2[0](out)
        out = self.layer2[1](out)
        out = self.layer3[0](out)
        out = self.layer3[1](out)
        out = self.layer4[0](out)
        out = self.layer4[1](out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def classifier1(self, x):
        out = self.fc(x) / self.temp
        return out

    def feature2(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out)
        out = self.layer1[1](out)
        out = self.layer2[0](out)
        out = self.layer2[1](out)
        out = self.layer3[0](out)
        out = self.layer3[1](out)
        out = self.layer4[0](out)
        return out

    def classifier2(self, x):
        out = self.layer4[1](x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        return out

    def feature3(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out)
        out = self.layer1[1](out)
        out = self.layer2[0](out)
        out = self.layer2[1](out)
        out = self.layer3[0](out)
        out = self.layer3[1](out)
        return out

    def classifier3(self, x):
        out = self.layer4[0](x)
        out = self.layer4[1](out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        return out


    def feature4(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out)
        out = self.layer1[1](out)
        out = self.layer2[0](out)
        out = self.layer2[1](out)
        out = self.layer3[0](out)
        return out

    def classifier4(self, x):
        out = self.layer3[1](x)
        out = self.layer4[0](out)
        out = self.layer4[1](out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        return out


    def feature5(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out)
        out = self.layer1[1](out)
        out = self.layer2[0](out)
        out = self.layer2[1](out)
        return out

    def classifier5(self, x):
        out = self.layer3[0](x)
        out = self.layer3[1](out)
        out = self.layer4[0](out)
        out = self.layer4[1](out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        return out
    '''
'''
class Linear(nn.Module):
    def __init__(self, num_classes=10, temp=1.0):
        super(Linear, self).__init__()
        self.fc_128 = nn.Linear(128, num_classes)
        self.fc_256 = nn.Linear(256, num_classes)
        self.fc_512 = nn.Linear(512, num_classes)
        self.fc_1024 = nn.Linear(1024, num_classes)
        self.fc_2048 = nn.Linear(2048, num_classes)
        self.fc_4096 = nn.Linear(4096, num_classes)
        self.fc_8192 = nn.Linear(8192, num_classes)
        self.fc_16384 = nn.Linear(16384, num_classes)
        self.fc_32768 = nn.Linear(32768, num_classes)
        self.fc_65536 = nn.Linear(65536, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc_32768(out)
        return out
'''



'''
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name,num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.avgpool2d = nn.AvgPool2d(kernel_size=1, stride=1)
        self.maxpoll2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


    def forward1(self, x):#11
        hiddens = []
        out = self.features[0](x)
        out = self.features[1](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[4](out)
        out = self.features[5](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[8](out)
        out = self.features[9](out)
        out = self.relu(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[11](out)
        out = self.features[12](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[15](out)
        out = self.features[16](out)
        out = self.relu(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[18](out)
        out = self.features[19](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[22](out)
        out = self.features[23](out)
        out = self.relu(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[25](out)
        out = self.features[26](out)
        out = self.relu(out)
        hiddens.append(out.view(out.size(0), -1))
        return hiddens

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(temp=1.0, **kwargs):
    model = VGG('VGG11', **kwargs)
    return model
'''

class Linear(nn.Module):
    def __init__(self, input_dim, num_classes=10, temp=1.0):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=10, temp=1.0):
        super(MLP, self).__init__()
        hidden_dim = 512
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc2(F.relu(self.fc1(out)))
        return out

def resnet18(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], temp=temp, **kwargs)
    return model


def resnet34(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], temp=temp, **kwargs)
    return model


def resnet50(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], temp=temp, **kwargs)
    return model


def resnet101(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], temp=temp, **kwargs)
    return model


def resnet110(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 26, 3], temp=temp, **kwargs)
    return model


def resnet152(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], temp=temp, **kwargs)
    return model




class ResNet_distillation(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, no_linear=False):
        super(ResNet_distillation, self).__init__()
        self.in_planes = 64
        self.no_linear = no_linear
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        hidden = out.view(out.size(0), -1)
        if self.no_linear:
            out = hidden
        else:
            out = self.linear(hidden)
        return out
    
class linearLayer(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearLayer, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

def resnet18_distillation(num_classes):
    return ResNet_distillation(BasicBlock, [2,2,2,2], num_classes)


class LeNet(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz = 28):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz//4
        self.n_feat = 50 * feat_map_sz * feat_map_sz

        # !!! [Architecture design tip] !!!
        # The KCL has much better convergence of optimization when the BN layers are added.
        # MCL is robust even without BN layer.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(500, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def lenet(num_classes):  # LeNet with color input
    return LeNet(out_dim=num_classes, in_channel=3, img_sz=32)






class DenseBottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(DenseBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def densenet121(**kwargs):
    return DenseNet(DenseBottleneck, [6,12,24,16], growth_rate=32, **kwargs)

def densenet169(**kwargs):
    return DenseNet(DenseBottleneck, [6,12,32,32], growth_rate=32, **kwargs)

def densenet201(**kwargs):
    return DenseNet(DenseBottleneck, [6,12,48,32], growth_rate=32, **kwargs)

def densenet161(**kwargs):
    return DenseNet(DenseBottleneck, [6,12,36,24], growth_rate=48, **kwargs)




cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name,num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.avgpool2d = nn.AvgPool2d(kernel_size=1, stride=1)
        self.maxpoll2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def forward1(self, x):#11
        hiddens = []
        out = self.features[0](x)
        out = self.features[1](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[4](out)
        out = self.features[5](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[8](out)
        out = self.features[9](out)
        out = self.relu(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[11](out)
        out = self.features[12](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[15](out)
        out = self.features[16](out)
        out = self.relu(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[18](out)
        out = self.features[19](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[22](out)
        out = self.features[23](out)
        out = self.relu(out)
        hiddens.append(out.view(out.size(0), -1))
        out = self.features[25](out)
        out = self.features[26](out)
        out = self.relu(out)
        hiddens.append(out.view(out.size(0), -1))
        return hiddens

    def forward1_(self, x):#11
        hiddens = []
        out = self.features[0](x)
        out = self.features[1](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out)
        out = self.features[4](out)
        out = self.features[5](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out)
        out = self.features[8](out)
        out = self.features[9](out)
        out = self.relu(out)
        hiddens.append(out)
        out = self.features[11](out)
        out = self.features[12](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out)
        out = self.features[15](out)
        out = self.features[16](out)
        out = self.relu(out)
        hiddens.append(out)
        out = self.features[18](out)
        out = self.features[19](out)
        out = self.relu(out)
        out = self.maxpoll2d(out)
        hiddens.append(out)
        out = self.features[22](out)
        out = self.features[23](out)
        out = self.relu(out)
        hiddens.append(out)
        out = self.features[25](out)
        out = self.features[26](out)
        out = self.relu(out)
        hiddens.append(out)
        return hiddens

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(temp=1.0, **kwargs):
    model = VGG('VGG11', **kwargs)
    return model



class Decoder(nn.Module):
    def __init__(self, inplanes, image_size, batch_size, interpolate_mode='bilinear', widen=1):
        super(Decoder, self).__init__()

        self.image_size = image_size

        assert interpolate_mode in ['bilinear', 'nearest']
        self.interpolate_mode = interpolate_mode

        self.bce_loss = nn.BCELoss()

        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes, int(12 * widen), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(12 * widen)),
            nn.ReLU(),
            nn.Conv2d(int(12 * widen), 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
            batch_size, 3, image_size, image_size
        ).cuda()
        self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
            batch_size, 3, image_size, image_size
        ).cuda()

    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]
        
    def forward(self, features, image_ori):
        image_ori = self._image_restore(image_ori)
        if self.interpolate_mode == 'bilinear':
            features = F.interpolate(features, size=[int(self.image_size), int(self.image_size)],
                                     mode='bilinear', align_corners=True)
        elif self.interpolate_mode == 'nearest':   # might be faster
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='nearest')
        else:
            raise NotImplementedError
        return self.bce_loss(self.decoder(features), image_ori)
