#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import os
import argparse
import math
from tqdm import tqdm
import numpy as np
import torch

from lib.models import model_factory
from configs import cfg_factory
from lib.logger import setup_logger
from lib.kitti_converted import get_data_loader
from tools import visualizer
import glob

import pdb

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_pth', type=str, default='../res/model_final.pth')
    parse.add_argument('--res_pth', dest='res_pth', type=str,  default='../res/pc')
    parse.add_argument('--model', dest='model', type=str, default='bisenetonpc2')
    return parse.parse_args()

@torch.no_grad()
def eval_model(net, batch_size, im_root, im_ann, num_cls):
    dl = get_data_loader(im_root, batch_size, listpath=im_ann, mode='val')
    net.eval()



def detect(cfg, weight_pth):
    # model
    net = model_factory[cfg.model_type](cfg.num_cls, output_aux=True)
    net.load_state_dict(torch.load(weight_pth))
    net.cuda()
    net.eval()

    # loop
    go = True
    while go:

        # data preprocess
        dl = glob.glob(cfg.test_data_path + '/*.npy')
        data = np.load(dl[0])
        img = data[:,:,:5].transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float).cuda()

        # predict
        logits = net(img)[0]
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # visualize
        out = visualizer.colorize_1c(np.array(preds.cpu()))

        visualizer.back_project(data[:,:,:3], out)



def main():
    args = parse_args()
    cfg = cfg_factory[args.model]
    detect(cfg, args.weight_pth)


if __name__ == "__main__":
    main()
