import numpy as np
import torch
from torchvision.models import resnet50,resnet101
import torch.nn as nn
import math

class resnet101_c(nn.Module):
    """"""
    def __init__(self):
        super(resnet101_c, self).__init__()
        resnet = resnet101(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.avgpool = nn.Identity()
        resnet.fc = nn.Identity()

        self.net = resnet
        # self.out_channels = [3, 64, 128, 256, 512, 512]
        self.out_channels = [3, 64, 256, 512, 1024, 2048]

    def forward(self, x):
        x1 = self.net.conv1(x)
        x1 = self.net.bn1(x1)
        x1 = self.net.relu(x1)
        x2 = self.net.maxpool(x1)
        x2 = self.net.layer1(x2)
        x3 = self.net.layer2(x2)
        x4 = self.net.layer3(x3)
        x5 = self.net.layer4(x4)

        side = [x1, x2, x3, x4, x5]

        return side

class resnet50_c(nn.Module):
    """"""

    def __init__(self):
        super(resnet50_c, self).__init__()
        resnet = resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.avgpool = nn.Identity()
        resnet.fc = nn.Identity()

        self.net = resnet
        # self.out_channels = [3, 64, 128, 256, 512, 512]
        self.out_channels = [3, 64, 256, 512, 1024, 2048]

    def forward(self, x):
        x1 = self.net.conv1(x)
        x1 = self.net.bn1(x1)
        x1 = self.net.relu(x1)
        x2 = self.net.maxpool(x1)
        x2 = self.net.layer1(x2)
        x3 = self.net.layer2(x2)
        x4 = self.net.layer3(x3)
        x5 = self.net.layer4(x4)

        side = [x1, x2, x3, x4, x5]

        return side


if __name__ == '__main__':
    model = resnet101_c()
    im = torch.zeros((1, 3, 100, 100))

    out = model(im)
    # for i in out:
    #     print(i.size())
    def get_model_parm_nums(model):
        total = sum([param.numel() for param in model.parameters()])
        total = float(total) / 1e6
        return total

    print(get_model_parm_nums(model))