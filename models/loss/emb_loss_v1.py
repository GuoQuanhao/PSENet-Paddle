# coding: utf-8
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class EmbLoss_v1(nn.Module):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v1, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = (1.0, 1.0)

    def forward_single(self, emb, instance, kernel, training_mask, bboxes):
        training_mask = (training_mask > 0.5).long()
        kernel = (kernel > 0.5).long()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).reshape([-1])
        instance = instance.reshape([-1])
        emb = emb.reshape([self.feature_dim, -1])

        unique_labels, unique_ids = paddle.unique(instance_kernel, return_index=True, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0

        emb_mean = emb.new_zeros((self.feature_dim, num_instance), dtype='float32')
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = paddle.mean(emb[:, ind_k], axis=1)

        l_agg = emb.new_zeros(num_instance, dtype='float32')  # bug
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, axis=0)
            dist = F.relu(dist - self.delta_v) ** 2
            l_agg[i] = paddle.mean(paddle.log(dist + 1.0))
        l_agg = paddle.mean(l_agg[1:])

        if num_instance > 2:
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).reshape([-1, self.feature_dim])
            # print(seg_band)

            mask = (1 - paddle.eye(num_instance, dtype='int32')).reshape([-1, 1]).repeat(1, self.feature_dim)
            mask = mask.reshape([num_instance, num_instance, -1])
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.reshape([num_instance * num_instance, -1])
            # print(mask)

            dist = emb_interleave - emb_band
            dist = dist[mask > 0].reshape([-1, self.feature_dim).norm(p=2, axis=1])
            dist = F.relu(2 * self.delta_d - dist) ** 2
            l_dis = paddle.mean(paddle.log(dist + 1.0))
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = paddle.mean(paddle.log(paddle.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb, instance, kernel, training_mask, bboxes, reduce=True):
        loss_batch = emb.new_zeros((emb.size(0)), dtype='float32')

        for i in range(loss_batch.size(0)):
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i], training_mask[i], bboxes[i])

        loss_batch = self.loss_weight * loss_batch

        if reduce:
            loss_batch = paddle.mean(loss_batch)

        return loss_batch
