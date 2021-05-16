import paddle

EPS = 1e-6

def iou_single(a, b, mask, n_class):
    valid = mask == 1
    a = a.masked_select(valid)
    b = b.masked_select(valid)
    miou = []
    for i in range(n_class):
        if a.shape == [0] and a.shape==b.shape:
            inter = paddle.to_tensor(0.0)
            union = paddle.to_tensor(0.0)
        else:
            inter = ((a == i).logical_and(b == i)).astype('float32')
            union = ((a == i).logical_or(b == i)).astype('float32')
        miou.append(paddle.sum(inter) / (paddle.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou

def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = a.reshape([batch_size, -1])
    b = b.reshape([batch_size, -1])
    mask = mask.reshape([batch_size, -1])

    iou = paddle.zeros((batch_size,), dtype='float32')
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = paddle.mean(iou)
    return iou
