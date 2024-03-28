import torch
import torch.nn as nn
import torch.nn.functional as F
# from lib.normalize import Normalize

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_planes=-1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if out_planes==-1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.conv2 = nn.Conv2d(planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        if out_planes==-1:
            pass

        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, out_planes=-1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if out_planes==-1:
            self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        else:
            self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        if out_planes==-1 or out_planes==2048:
            pass
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, low_dim=16, args = None):
        super(ResNet, self).__init__()
        
        self.arch = "resnet34"
        self.avg_size = 1
        self.in_planes = 64

        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.avg_size = int(self.avg_size)
        self.avgpool = nn.AdaptiveAvgPool2d((self.avg_size, self.avg_size))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_last(block, 512, num_blocks[3], stride=2, out_planes=512)
        self.fc2 = nn.Linear(512*self.avg_size*self.avg_size*block.expansion, 256*4)
       
        if self.arch == "resnet34":
            self.fc = nn.Linear(512*self.avg_size*self.avg_size*block.expansion, low_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_last(self, block, planes, num_blocks, stride, out_planes=-1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides[:-1]:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        layers.append(block(self.in_planes, planes, strides[-1], out_planes=out_planes))
        self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out1 = self.fc(out)
        out2 = self.fc2(out)
        
        return out1,out2


def ResNet18(low_dim=16, args = None,point_num=16):
    return ResNet(BasicBlock, [2,2,2,2], low_dim=point_num * 2, args=args)

def ResNet34(low_dim=16, args = None,point_num=16):
    return ResNet(BasicBlock, [3,4,6,3], low_dim=point_num * 2, args=args)

def ResNet50(low_dim=4, args = None,point_num=16):
    return ResNet(Bottleneck, [3,4,6,3], low_dim=point_num * 2, args=args)

def ResNet101(low_dim=16, args = None,point_num=16):
    return ResNet(Bottleneck, [3,4,23,3], low_dim=point_num * 2, args=args)

def ResNet152(low_dim=16, args = None,point_num=16):
    return ResNet(Bottleneck, [3,8,36,3], low_dim=point_num * 2, args=args)

def test():
    net = ResNet101()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

