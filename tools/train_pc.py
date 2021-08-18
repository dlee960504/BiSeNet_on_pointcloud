#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
#sys.path.insert(0, '.')
import os
import os.path as osp
sys.path.append('..')

import random
import logging
import time
import argparse
from tqdm import tqdm
import numpy as np
from tabulate import tabulate
import yaml

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from lib.models import model_factory
from configs import cfg_factory
from lib.kitti_converted import get_data_loader
from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg

from datasets.semanticKITTI import parser

# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False


## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parse = argparse.ArgumentParser()
    #parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    #parse.add_argument('--port', dest='port', type=int, default=44554,)
    parse.add_argument('--model', dest='model', type=str, default='bisenetonpc2',)
    parse.add_argument('--data', default='../datasets/semanticKITTI', help='specify where data and data config is')
    parse.add_argument('--pth_dir', required=False, help='dir to pth file from which training will resume')
    parse.add_argument('--start_epoch', default=0, type=int, required=False, help='starting epoch if fine tuning')
    #parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = cfg_factory[args.model]

def set_model():
    net = model_factory[cfg.model_type](cfg.num_cls)
    if not args.pth_dir is None:
         net.load_state_dict(torch.load(args.pth_dir, map_location='cuda'))
    if cfg.use_sync_bn: net = set_syncbn(net)
    net.to(dtype=torch.float)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux

def set_syncbn(net):
    if has_apex:
        net = parallel.convert_syncbn_model(net)
    else:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_model_dist(net):
    if has_apex:
        net = parallel.DistributedDataParallel(net, delay_allreduce=True)
    else:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank)
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters

## @brief get parser for semantinc KITTI datset
def create_parser():
    try:
        print('opening data conifg')
        DATA = yaml.safe_load(open(args.data + '/config/data_cfg.yaml', 'r'))
    except Exception as e:
        print(e)
        exit()

    datadir = args.data
    data_parser = parser.Parser(root=datadir, 
                    train_sequences=DATA['split']['train'], 
                    valid_sequences=None, 
                    test_sequences=None, 
                    labels=DATA['labels'],
                    color_map=DATA['color_map'],
                    learning_map=DATA['learning_map'],
                    learning_map_inv=DATA['learning_map_inv'],
                    sensor=DATA['dataset']['sensor'],
                    max_points=DATA['dataset']['max_points'],
                    batch_size=5,
                    workers=0,
                    gt=True,
                    shuffle_train=False
                    )
    
    return data_parser


def train():
    logger = logging.getLogger()
    is_dist = dist.is_initialized()

    ## dataset
    #dl = get_data_loader(cfg.im_root, cfg.batch_size, listpath=cfg.train_im_anns, max_iter=cfg.max_iter, mode='train')
    data_parser = create_parser()
    dl = data_parser.trainloader

    ## model
    net, criteria_pre, criteria_aux = set_model()

    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex:
        opt_level = 'O1' if cfg.use_fp16 else 'O0'
        net, optim = amp.initialize(net, optim, opt_level=opt_level)

    ## ddp training
    # net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loop
    epoch_iter = args.start_epoch
    step = 0
    display_term = 500
    save_term = 2

    # iteration criterion: while step <= cfg.max_iter 
    while epoch_iter <= cfg.max_epoch:
        try:
            logger.info('epoch {} started...'.format(epoch_iter))
            diter = enumerate(tqdm(dl))
            for it, sample in diter:
                
                im = sample['img'].cuda()
                lb = sample['label'].cuda()

                lb = torch.squeeze(lb, 1)

                optim.zero_grad()
                logits, *logits_aux = net(im)
                loss_pre = criteria_pre(logits, lb)
                loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
                loss = loss_pre + sum(loss_aux)
                if has_apex:
                    with amp.scale_loss(loss, optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optim.step()
                torch.cuda.synchronize()
                lr_schdr.step()

                time_meter.update()
                loss_meter.update(loss.item())
                loss_pre_meter.update(loss_pre.item())
                _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
                
                ## update iterator
                step += 1

            ## print training log message
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(
                step, cfg.max_iter, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)

            if epoch_iter % save_term == 0 and epoch_iter != 0:
                save_pth = osp.join(cfg.respth,'model_epoch_{}.pth'.format(epoch_iter))
                logger.info('\nsave models to {}'.format(save_pth))
                state = net.state_dict()
                torch.save(state, save_pth)

            ## upate iterator
            epoch_iter += 1
        
        except RuntimeError as e:
            #print('diminishing gradient problem. finish the training')
            print(e)
            break

    ## dump the final model and evaluate the result
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.state_dict()
    torch.save(state, save_pth)
    #if dist.get_rank() == 0: torch.save(state, save_pth)

    #logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    #heads, mious = eval_model(net, 2, cfg.im_root, cfg.val_im_anns)
    #logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

    return


def main():
    #torch.cuda.set_device(args.local_rank)
    #dist.init_process_group( backend='nccl', init_method='tcp://127.0.0.1:{}'.format(args.port), world_size=torch.cuda.device_count(), rank=args.local_rank)
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-train'.format(cfg.model_type), cfg.respth)
    train()


if __name__ == "__main__":
    main()
