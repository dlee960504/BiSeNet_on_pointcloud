#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
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

from lib.models import model_factory
from configs import cfg_factory
from lib.logger import setup_logger
from lib.kitti_converted import get_data_loader
from tools import visualizer


def parse_args():
    parse = argparse.ArgumentParser()
    #parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--weight-path', dest='weight_pth', type=str, default='../res/model_final.pth',)
    #parse.add_argument('--port', dest='port', type=int, default=44553,)
    parse.add_argument('--model', dest='model', type=str, default='bisenetonpc',)
    return parse.parse_args()

@torch.no_grad()
def eval_model(net, batch_size, im_root, im_ann, num_cls):
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

    ## evaluator
    heads, mious = eval_model(net, 2, cfg.val_im_root, cfg.val_im_anns, cfg.num_cls)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

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
