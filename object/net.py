import torch.nn as nn
import torch.nn.functional as F


def flatten_conv(x, k):
    """
    Flatten the convolutional layer
    :param x:
    :param k:
    :return:
    """
    bs, nf, gx, gy = x.size()
    x = x.permute(0, 2, 3, 1).contiguous()
    return x.view(bs, -1, nf//k)


class ConvBlock(nn.Module):
    """
    This is a convolutional layer block that is used as a 2d convolutional unit
    """
    def __init__(self, nin, nout, stride=2, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)

    def forward(self, x): return self.drop(self.bn(F.relu(self.conv(x))))


class Output(nn.Module):
    """
    This is the network output. Note that there are two different output layers for regression
    and classification that share the same convolutional block input.
    """
    def __init__(self, k, nin, bias, num_classes):
        super().__init__()
        self.k = k
        # Classification layer
        self.output_classification = nn.Conv2d(nin, (num_classes+1)*k, 3, padding=1)
        # Bounding box regression layer
        self.output_regression = nn.Conv2d(nin, 4*k, 3, padding=1)
        self.output_classification.bias.data.zero_().add(bias)

    def forward(self, x):
        return [
            flatten_conv(self.output_classification(x), self.k),
            flatten_conv(self.output_regression(x), self.k)
        ]


class SSDHead(nn.Module):
    """
    The SSD extension of ResNet34
    """
    def __init__(self, k, bias, num_classes):
        super().__init__()
        self.drop = nn.Dropout(0.25)
        self.first_convolution = ConvBlock(512, 256, stride=1)
        self.second_convolution = ConvBlock(256, 256)
        self.out = Output(k, 256, bias, num_classes)

    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.first_convolution(x)
        x = self.second_convolution(x)
        return self.out(x)
