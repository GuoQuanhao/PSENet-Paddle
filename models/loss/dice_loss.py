import paddle
import paddle.nn as nn


class DiceLoss(nn.Layer):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.shape[0]

        input = paddle.nn.functional.sigmoid(input)

        input = input.reshape([batch_size, -1])
        target = target.reshape([batch_size, -1]).astype('float32')
        mask = mask.reshape([batch_size, -1]).astype('float32')


        input = input * mask
        target = target * mask

        a = paddle.sum(input * target, axis=1)
        b = paddle.sum(input * input, axis=1) + 0.001
        c = paddle.sum(target * target, axis=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d
        loss = self.loss_weight * loss
        if reduce:
            loss = paddle.mean(loss)

        return loss
