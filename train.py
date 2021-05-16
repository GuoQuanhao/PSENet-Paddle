import paddle
import numpy as np
import random
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from utils import AverageMeter, DataLoader

paddle.seed(123456)
np.random.seed(123456)
random.seed(123456)


def train(train_loader, model, optimizer, epoch, start_iter, cfg, len_train_loader):
    model.train()

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()
    accs_rec = AverageMeter()

    # start time
    start = time.time()
    for iter, data in enumerate(train_loader()):
        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            sys.stdout.flush()
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data.update(dict(cfg=cfg))

        # forward
        outputs = model(**data)
        #
        # print(outputs['loss_text'].shape)
        # print(outputs['loss_kernels'].shape)

        # detection loss
        loss_text = paddle.mean(outputs['loss_text'])
        losses_text.update(loss_text.numpy())

        loss_kernels = paddle.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.numpy())

        loss = loss_text + loss_kernels

        iou_text = paddle.mean(outputs['iou_text'])
        ious_text.update(iou_text.numpy())
        iou_kernel = paddle.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.numpy())

        losses.update(loss.numpy())
        # backward
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 20 == 0:
            output_log = '({batch}/{size}) LR: {lr:.6f} | Batch: {bt:.3f}s | Total: {total:.0f}min | ' \
                         'ETA: {eta:.0f}min | Loss: {loss:.3f} | ' \
                         'Loss(text/kernel): {loss_text:.3f}/{loss_kernel:.3f} ' \
                         '| IoU(text/kernel): {iou_text:.3f}/{iou_kernel:.3f} | Acc rec: {acc_rec:.3f}'.format(
                batch=iter + 1,
                size=len_train_loader,
                lr=optimizer.get_lr(),
                bt=batch_time.avg,
                total=batch_time.avg * iter / 60.0,
                eta=batch_time.avg * (len_train_loader - iter) / 60.0,
                loss_text=float(losses_text.avg),
                loss_kernel=float(losses_kernels.avg),
                loss=float(losses.avg),
                iou_text=float(ious_text.avg),
                iou_kernel=float(ious_kernel.avg),
                acc_rec=accs_rec.avg,
            )
            print(output_log)
            sys.stdout.flush()


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule
    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1
    optimizer._learning_rate = lr


def save_checkpoint(state, checkpoint_path, cfg):
    weight_file_path = osp.join(checkpoint_path, 'checkpoint.pdparams')
    optimizer_file_path = osp.join(checkpoint_path, 'checkpoint.pdopt')
    paddle.save(state['state_dict'], weight_file_path)
    paddle.save(state['optimizer'], optimizer_file_path)

    if cfg.data.train.type in ['synth'] or \
            (state['iter'] == 0 and state['epoch'] > cfg.train_cfg.epoch - 100 and state['epoch'] % 10 == 0):
        weight_file_name = 'checkpoint_%dep.pdparams' % state['epoch']
        weight_file_path = osp.join(checkpoint_path, weight_file_name)
        optimizer_file_name = 'checkpoint_%dep.pdopt' % state['epoch']
        optimizer_file_path = osp.join(checkpoint_path, optimizer_file_name)
        paddle.save(state['state_dict'], weight_file_path)
        paddle.save(state['optimizer'], optimizer_file_path)


def main(args):
    cfg = Config.fromfile(args.config)
    print(json.dumps(cfg._cfg_dict, indent=4))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('checkpoints', cfg_name)
    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    print('Checkpoint path: %s.' % checkpoint_path)
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.train)
    train_loader = DataLoader(
        data_loader,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=True,
    )

    len_train_loader = int(len(data_loader) // cfg.data.batch_size) + 1

    # model
    model = build_model(cfg.model)
    # model = paddle.DataParallel(model)

    # Check if model has custom optimizer / loss
    if hasattr(model.sublayers, 'optimizer'):
        optimizer = model.sublayers.optimizer
    else:
        if cfg.train_cfg.optimizer == 'SGD':
            optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=cfg.train_cfg.lr, weight_decay=5e-4)
        elif cfg.train_cfg.optimizer == 'Adam':
            optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=cfg.train_cfg.lr)

    start_epoch = 531
    start_iter = 0
    if hasattr(cfg.train_cfg, 'pretrain'):
        assert osp.isfile(cfg.train_cfg.pretrain), 'Error: no pretrained weights found!'
        print('Finetuning from pretrained model %s.' % cfg.train_cfg.pretrain)
        checkpoint = paddle.load(cfg.train_cfg.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
    if args.resume:
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        print('Resuming from checkpoint %s.' % args.resume)
        checkpoint = paddle.load(args.resume)
        model.set_state_dict(checkpoint)
        optimizer.set_state_dict(args.resume.split('.')[0] + '.pdopt')

    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))

        train(train_loader, model, optimizer, epoch, start_iter, cfg, len_train_loader)

        state = dict(
            epoch=epoch + 1,
            iter=0,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        save_checkpoint(state, checkpoint_path, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    args = parser.parse_args()

    main(args)
