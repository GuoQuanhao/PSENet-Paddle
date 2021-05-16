import paddle
import argparse
import os
import os.path as osp
import sys
import json
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import ResultFormat, AverageMeter, DataLoader


def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def test(data_loader, model, cfg):
    model.eval()

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_pse_time=AverageMeter(500),
            rec_time=AverageMeter(500),
            total_time=AverageMeter(500)
        )

    for idx, data in enumerate(data_loader):
        print('Testing %d/%d' % (idx+1, len(data_loader)))
        sys.stdout.flush()

        # print(i, data['imgs'].unsqueeze(0).shape, paddle.to_tensor(data['img_metas']['org_img_size']).reshape([1, -1]).shape, paddle.to_tensor(data['img_metas']['img_size']).reshape([1, -1]).shape)
        # prepare input
        data['imgs'] = data['imgs'].unsqueeze(0)
        data['img_metas']['org_img_size'] = paddle.to_tensor(data['img_metas']['org_img_size']).reshape([1, -1])
        data['img_metas']['img_size'] = paddle.to_tensor(data['img_metas']['img_size']).reshape([1, -1])

        data.update(dict(
            cfg=cfg
        ))
        # forward
        with paddle.no_grad():
            outputs = model(**data)

        if cfg.report_speed:
            report_speed(outputs, speed_meters)

        # save result
        image_name, _ = osp.splitext(osp.basename(data_loader.img_paths[idx]))
        # print('image_name', image_name)
        rf.write_result(image_name, outputs)


def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.test)

    # model
    model = build_model(cfg.model)
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            sys.stdout.flush()

            checkpoint = paddle.load(args.checkpoint)

            d = dict()
            for key, value in checkpoint.items():
                d[key] = value
            model.set_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint))

    # fuse conv and bn
    model = fuse_module(model)

    # test
    test(data_loader, model, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)
