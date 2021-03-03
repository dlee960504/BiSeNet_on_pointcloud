#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
sys.path.append('..')
import os
import logging
import argparse
import math
from tabulate import tabulate

from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lib.models import model_factory
from configs import cfg_factory
from lib.logger import setup_logger
from lib.kitti_converted import get_data_loader



class MscEvalV0(object):

    def __init__(self, scales=(0.5, ), flip=False, ignore_label=255):
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, sample in diter:
            imgs = sample['img'].cuda()
            label = sample['label']
            label = torch.unsqueeze(label, 1)
            N, _, H, W = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            probs = torch.zeros(
                    (N, n_classes, H, W), dtype=torch.float32).cuda().detach()
            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                logits = net(im_sc)[0]
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc)[0]
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()

@torch.no_grad()
def eval_model(net, batch_size, im_root, im_ann, num_cls):
    is_dist = dist.is_initialized()
    dl = get_data_loader(im_root, batch_size, listpath=im_ann, mode='val')
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0((1., ), False)
    mIOU = single_scale(net, dl, num_cls)
    heads.append('single_scale')
    mious.append(mIOU)
    logger.info('single mIOU is: %s\n', mIOU)    

    return heads, mious


def evaluate(cfg, weight_pth):
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = model_factory[cfg.model_type](cfg.num_cls, output_aux=True)
    #  net = BiSeNetV2(19)
    net.load_state_dict(torch.load(weight_pth))
    net.cuda()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank, ], output_device=local_rank)

    ## evaluator
    heads, mious = eval_model(net, 2, cfg.val_im_root, cfg.val_im_anns, cfg.num_cls)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))


def parse_args():
    parse = argparse.ArgumentParser()
    #parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--weight-path', dest='weight_pth', type=str, default='../res/model_final.pth',)
    #parse.add_argument('--port', dest='port', type=int, default=44553,)
    parse.add_argument('--model', dest='model', type=str, default='bisenetonpc',)
    return parse.parse_args()


def main():
    args = parse_args()
    cfg = cfg_factory[args.model]
    #if not args.local_rank == -1:
    #    torch.cuda.set_device(args.local_rank)
    #    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(args.port), world_size=torch.cuda.device_count(), rank=args.local_rank)
    if not os.path.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-eval'.format(cfg.model_type), cfg.respth)
    evaluate(cfg, args.weight_pth)


if __name__ == "__main__":
    main()
