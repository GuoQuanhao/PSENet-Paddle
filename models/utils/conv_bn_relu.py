import paddle
import paddle.nn as nn
import math


class Conv_BN_ReLU(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
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
