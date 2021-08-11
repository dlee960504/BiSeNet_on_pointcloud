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

import time



class MscEvalV0(object):

    def __init__(self, scales=(0.5, ), flip=False, ignore_label=255, use_cpu=False):
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label
        self.use_cpu = use_cpu

    def __call__(self, net, dl, n_classes):
        ## evaluate
        if not self.use_cpu:
            hist = torch.zeros(n_classes, n_classes).cuda().detach()
        else:
            hist = torch.zeros(n_classes, n_classes).detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        
        durations = []

        for i, sample in diter:
            # timer
            start_time = time.time()

            # sample parsing
            if not self.use_cpu:
                imgs = sample['img'].cuda()
            else:
                imgs = sample['img']
            label = sample['label']
            label = torch.unsqueeze(label, 1)
            N, _, H, W = label.shape
            if not self.use_cpu:
                label = label.squeeze(1).cuda()
            else:
                label = label.squeeze(1)
            size = label.size()[-2:]
            if not self.use_cpu:
                probs = torch.zeros((N, n_classes, H, W), dtype=torch.float32).cuda().detach()
            else:
                probs = torch.zeros((N, n_classes, H, W), dtype=torch.float32).detach()
            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)
                if not self.use_cpu:
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

            import pdb; pdb.set_trace()

            # timer
            duration = time.time() - start_time
            durations.append(duration)

            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious[1:].mean()

        heads = ['Car', 'Pedestrian', 'Cyclist', 'mIOU', 'avg_duration']
        data = ious.tolist()[1:]
        data.append(miou.item())
        data.append(np.mean(durations))
        output_info1 = tabulate([data,], headers=heads, tablefmt='orgtbl')
        print(output_info1 + '\n')

        return miou.item()

@torch.no_grad()
def eval_model(net, batch_size, im_root, im_ann, num_cls, use_cpu=False):
    is_dist = dist.is_initialized()
    dl = get_data_loader(im_root, batch_size, listpath=im_ann, mode='val')
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0((1., ), flip=False, use_cpu=use_cpu)
    mIOU = single_scale(net, dl, num_cls)
    heads.append('single_scale')
    mious.append(mIOU)
    logger.info('single mIOU is: %s\n', mIOU)    

    return heads, mious


def evaluate(cfg, weight_pth, use_cpu=False):
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = model_factory[cfg.model_type](cfg.num_cls, output_aux=True)
    #  net = BiSeNetV2(19)
    
    if not use_cpu:
        net.load_state_dict(torch.load(weight_pth))
        net.cuda()
    else:
        net.load_state_dict(torch.load(weight_pth, map_location=torch.device('cpu')))
        net.cpu()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank, ], output_device=local_rank)

    ## evaluator
    heads, mious = eval_model(net, 2, cfg.val_im_root, cfg.val_im_anns, cfg.num_cls, use_cpu)
    output_info = tabulate([mious, ], headers=heads, tablefmt='orgtbl')
    logger.info(output_info)

    # print is for console and IO redirectioning
    print(output_info + '\n')

def evaluate_ssgv3():
    pred_path = '../../SqueezeSegV3/res/'
    ans_path = '../datasets/Kitti_test/npydata/'
    
    import glob
    pred_list = sorted(glob.glob(pred_path + '*.npy'))
    ans_list = sorted(glob.glob(ans_path + '*.npy'))

    # dummy duration
    durations = []

    #tensor setting
    n_classes = 4
    hist = torch.zeros(n_classes, n_classes).detach()

    for pred, ans in zip(pred_list, ans_list):

        pred_name = os.path.basename(pred).split('_pred.npy')[0]
        ans_name = os.path.basename(ans).split('.npy')[0]

        assert pred_name == ans_name, 'pred_name: {}, ans_name: {} --> not matching answer'.format(pred_name, ans_name)

        ignore_label=255

        # remapping
        pred = np.load(pred)

        #import pdb; pdb.set_trace()

        pred = np.where(pred==2, 3, pred) # cyclist
        pred = np.where(pred==6, 2, pred) # person
        #pred = np.where(np.any([pred==4, pred==5], axis=0), 1, pred) # other vehicle, truck --> car
        pred = np.where(pred==4, 0, pred)
        pred = np.where(np.any([pred==7, pred==8], axis=0), 3, pred) # bicyclist, motorcyclist --> cyclist
        pred = np.where(np.any([pred==9, np.all([pred>=11, pred<=19], axis=0)], axis=0), 0, pred) # road, sidewalk, other-ground, building, fence, vegetation, trunk, terrain, pole, traffic-sign  --> unlabeled
        pred = np.where(pred==10, 0, pred) # parking --> unlabeled
        pred = np.where(pred==5, 0, pred)

        #pdb.set_trace()

        label = np.load(ans)[:,:,5]
        keep = label != ignore_label

        #duration = time.time() - start_time
        durations.append(0)

        pred = torch.from_numpy(pred.astype(np.int32))
        label = torch.from_numpy(label.astype(np.int32))

        hist += torch.bincount(label[keep] * n_classes + pred[keep],minlength=n_classes ** 2).view(n_classes, n_classes)
        
    ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
    miou = ious[1:].mean()

    heads = ['Car', 'Pedestrian', 'Cyclist', 'mIOU', 'avg_duration']
    data = ious.tolist()[1:]
    data.append(miou.item())
    data.append(np.mean(durations))
    output_info1 = tabulate([data,], headers=heads, tablefmt='orgtbl')
    print(output_info1 + '\n')

    return miou.item()


def parse_args():
    parse = argparse.ArgumentParser()
    #parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--weight_path', dest='weight_pth', type=str, default='../res/model_final.pth',)
    #parse.add_argument('--port', dest='port', type=int, default=44553,)
    parse.add_argument('--model', dest='model', type=str, default='bisenetonpc2',)
    parse.add_argument('--cpu', action='store_true', default=False, required=False)
    return parse.parse_args()


def main():
    args = parse_args()
    cfg = cfg_factory[args.model]
    use_cpu = args.cpu
    #if not args.local_rank == -1:
    #    torch.cuda.set_device(args.local_rank)
    #    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(args.port), world_size=torch.cuda.device_count(), rank=args.local_rank)
    if not os.path.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-eval'.format(cfg.model_type), cfg.respth)
    evaluate(cfg, args.weight_pth, use_cpu)
    #evaluate_ssgv3()


if __name__ == "__main__":
    main()
