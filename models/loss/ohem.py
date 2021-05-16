import paddle
import numpy as np


def ohem_single(score, gt_text, training_mask):
    pos_num = np.sum((gt_text > 0.5).numpy()) - np.sum((gt_text > 0.5).numpy() & (training_mask <= 0.5).numpy())

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape([1, selected_mask.shape[0], selected_mask.shape[1]]).astype('float32')
        return selected_mask

    neg_num = np.sum((gt_text <= 0.5).numpy())
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape([1, selected_mask.shape[0], selected_mask.shape[1]]).astype('float32')
        return selected_mask

    neg_score = score.masked_select(gt_text <= 0.5)
    neg_score_sorted = paddle.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold).logical_or(gt_text > 0.5)).logical_and(training_mask > 0.5)
    selected_mask = selected_mask.reshape([1, selected_mask.shape[0], selected_mask.shape[1]]).astype('float32')
    return selected_mask

def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = paddle.concat(selected_masks, 0).astype('float32')
    return selected_masks
