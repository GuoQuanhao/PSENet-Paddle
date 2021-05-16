import os
import sys
import paddle
import paddle.nn as nn
import math
from paddle.utils.download import get_weights_path_from_url

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


__all__ = ['resnet18', 'resnet50', 'resnet101']

model_urls = {
    'resnet18': 'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
    'resnet50': 'https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
    'resnet101': 'https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams'
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes, momentum=0.1)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes, momentum=0.1)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes, momentum=0.1)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Convkxk(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Convkxk, self).__init__()
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias_attr=False)
        self.bn = nn.BatchNorm2D(out_planes, momentum=0.1)
        self.relu = nn.ReLU()

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.Normal(0, math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNet(nn.Layer):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2D(64, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2D(64, momentum=0.1)
        self.relu2 = nn.ReLU()
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2D(128, momentum=0.1)
        self.relu3 = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.Normal(0, math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        f = []
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        x = self.layer4(x)
        f.append(x)

        return tuple(f)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.set_state_dict(load_url(model_urls['resnet18']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.set_state_dict(load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.set_state_dict(load_url(model_urls['resnet101']))
    return model


def load_url(url):
    local_weight_path = get_weights_path_from_url(url)
    pretrain_dict = paddle.load(local_weight_path)
    return pretrain_dict

