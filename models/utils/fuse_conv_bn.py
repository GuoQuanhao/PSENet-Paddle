import paddle
import paddle.nn as nn


class Identity(nn.Layer):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = paddle.randn(128, 20)
        >>> output = m(input)
        >>> print(output.shape)
        [128, 20]

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return input


def fuse_conv_bn(conv, bn):
    """During inference, the functionary of batch norm layers is turned off but
    only the mean and var alone channels are used, which exposes the chance to
    fuse it with the preceding conv layers to save computations and simplify
    network structures."""
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else paddle.zeros_like(
        bn._mean)

    factor = bn.weight / paddle.sqrt(bn._variance + bn._epsilon)
    conv.weight = paddle.create_parameter(shape=conv.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.Assign(conv_w * factor.reshape([conv._out_channels, 1, 1, 1])))
    conv.bias = paddle.create_parameter(shape=((conv_b - bn._mean) * factor + bn.bias).shape, dtype='float32', default_initializer=paddle.nn.initializer.Assign((conv_b - bn._mean) * factor + bn.bias))
    return conv


def fuse_module(m):
    last_conv = None
    last_conv_name = None

    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2D, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = fuse_conv_bn(last_conv, child)
            m._sub_layers[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            m._sub_layers[name] = Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2D):
            last_conv = child
            last_conv_name = name
        else:
            fuse_module(child)
    return m
