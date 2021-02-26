#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path
import json

import torch
from torch.utils.data import Dataset, DataLoader
#import torch.distributed as dist
import numpy as np

#import lib.transform_cv2 as T
#from lib.sampler import RepeatedDistSampler
#from lib.base_dataset import BaseDataset, TransformationTrain, TransformationVal


labels_info_KITTI = [
    {"hasInstances": False, "category": "ground", "catid": 0, "name": "road", "ignoreInEval": True, "id": 40, "color": [255, 0, 255], "trainId": 255},
    {"hasInstances": False, "category": "ground", "catid": 0, "name": "sidewalk", "ignoreInEval": True, "id": 48, "color": [75, 0, 75], "trainId": 255},
    {"hasInstances": False, "category": "ground", "catid": 0, "name": "parking", "ignoreInEval": True, "id": 44, "color": [255, 150, 255], "trainId": 255},
    {"hasInstances": False, "category": "ground", "catid": 0, "name": "other-ground", "ignoreInEval": True, "id": 49, "color": [75, 0, 175], "trainId": 255},
    {"hasInstances": False, "category": "strcuture", "catid": 0, "name": "building", "ignoreInEval": True, "id": 50, "color": [0, 200, 255], "trainId": 255},
    {"hasInstances": False, "category": "structure", "catid": 0, "name": "other-structure", "ignoreInEval": True, "id": 52, "color": [0, 150, 255], "trainId": 255},
    {"hasInstances": False, "category": "vehicle", "catid": 1, "name": "car", "ignoreInEval": True, "id": 10, "color": [245, 150, 100], "trainId": 255},
    {"hasInstances": False, "category": "vehicle", "catid": 1, "name": "truck", "ignoreInEval": True, "id": 18, "color": [180, 30, 80], "trainId": 255},
    {"hasInstances": False, "category": "vehicle", "catid": 3, "name": "bicycle", "ignoreInEval": True, "id": 11, "color": [245, 150, 100], "trainId": 255},
    {"hasInstances": False, "category": "vehicle", "catid": 3, "name": "motorcycle", "ignoreInEval": True, "id": 15, "color": [150, 60, 30], "trainId": 255},
    {"hasInstances": False, "category": "vehicle", "catid": 1, "name": "other-vehicle", "ignoreInEval": True, "id": 20, "color": [255, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "nature", "catid": 0, "name": "vegetation", "ignoreInEval": True, "id": 70, "color": [0, 175, 0], "trainId": 255},
    {"hasInstances": False, "category": "nature", "catid": 0, "name": "trunk", "ignoreInEval": True, "id": 71, "color": [0, 60, 135], "trainId": 255},
    {"hasInstances": False, "category": "nature", "catid": 0, "name": "terrain", "ignoreInEval": True, "id": 72, "color": [80, 240, 150], "trainId": 255},
    {"hasInstances": False, "category": "human", "catid": 2, "name": "person", "ignoreInEval": True, "id": 30, "color": [30, 30, 255], "trainId": 255},
    {"hasInstances": False, "category": "human", "catid": 3, "name": "bicylist", "ignoreInEval": True, "id": 31, "color": [200, 40, 255], "trainId": 255},
    {"hasInstances": False, "category": "human", "catid": 3, "name": "motorcyclist", "ignoreInEval": True, "id": 32, "color": [90, 30, 150], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 0, "name": "fence", "ignoreInEval": True, "id": 51, "color": [50, 120, 255], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 0, "name": "pole", "ignoreInEval": True, "id": 80, "color": [150, 240, 255], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 0, "name": "traffic-sign", "ignoreInEval": True, "id": 81, "color": [0, 0, 255], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 0, "name": "other-object", "ignoreInEval": True, "id": 99, "color": [255, 255, 50], "trainId": 255},
    {"hasInstances": False, "category": "moving", "catid": 1, "name": "moving-car", "ignoreInEval": True, "id": 252, "color": [245, 150, 100], "trainId": 255},
    {"hasInstances": False, "category": "moving", "catid": 3, "name": "moving-bicyclist", "ignoreInEval": True, "id": 253, "color": [200, 40, 255], "trainId": 255},
    {"hasInstances": False, "category": "moving", "catid": 2, "name": "moving-person", "ignoreInEval": True, "id": 254, "color": [30, 30, 255], "trainId": 255},
    {"hasInstances": False, "category": "moving", "catid": 3, "name": "moving-motorcyclist", "ignoreInEval": True, "id": 255, "color": [90, 30, 150], "trainId": 255},
    {"hasInstances": False, "category": "moving", "catid": 0, "name": "moving-on-rails", "ignoreInEval": True, "id": 256, "color": [255, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "moving", "catid": 1, "name": "moving-bus", "ignoreInEval": True, "id": 257, "color": [250, 80, 100], "trainId": 255},
    {"hasInstances": False, "category": "moving", "catid": 1, "name": "moving-truck", "ignoreInEval": True, "id": 258, "color": [180, 30, 80], "trainId": 255},
    {"hasInstances": False, "category": "moving", "catid": 1, "name": "moving-other-vehicle", "ignoreInEval": True, "id": 259, "color": [255, 0, 0], "trainId": 255},
]

labels_info = [{"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [255, 255, 255], "trainId": 255},
               {"hasInstances": False, "category": "void", "catid": 1, "name": "car", "ignoreInEval": False, "id": 1, "color": [0, 0, 255], "trainId": 1},
               {"hasInstances": False, "category": "void", "catid": 2, "name": "pedestrian", "ignoreInEval": False, "id": 2, "color": [255, 0, 0], "trainId": 2}
               {"hasInstances": False, "category": "void", "catid": 3, "name": "cyclist", "ignoreInEval": False, "id": 3, "color": [0, 255, 0], "trainId": 3}]


class KITTIconverted(Dataset):
    '''
    '''
    def __init__(self, root_path, list_path=None, trans_func=None, mode='train'):
        super(KITTIconverted, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.trans_func = trans_func

        self.n_cats = 19
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)

        self._to_tensor = ToTensor()

        self.data_paths = []
        if list_path == None:
            list_path = root_path + '/data_list.txt'
        with open(list_path, 'r') as f:
            npy_names = f.read().splitlines()
            for npy_name in npy_names:
                self.data_paths.append(os.path.join(root_path, npy_name))

        #for elem in labels_info:
            #self.lb_map[elem['id']] = elem['catid']

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = self.data_paths[idx]
        data = np.load(data_path)
        img, label = data[:,:,:5], data[:,:,5]
        #print('inside getitem: ', img.shape, label.shape)
        sample = {'img': img, 'label': label}
        if not self.trans_func is None:
            sample = self.trans_func(sample)
        sample = self._to_tensor(sample)
        #img, label = img_lab['img'], img_lab['label']
        #return img.detach(), label.detach()
        return sample

def get_data_loader(datapath, batchsize, listpath=None, max_iter=None, mode='train', distributed=False):
    if mode == 'train':
        #trans_func = TransformationTrain(scales, cropsize)
        trans_func = None
        shuffle = True
        drop_last = True
    elif mode == 'val':
        #trans_func = TransformationVal()
        trans_func = None
        shuffle = False
        drop_last = False

    ds = KITTIconverted(datapath, list_path=listpath, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = ims_per_gpu * dist.get_world_size() * max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['img'], sample['label']
        image = image.transpose((2, 0, 1))
        return {'img': torch.tensor(image), 'label': torch.tensor(label, dtype=torch.long)}


if __name__ == "__main__":
    #from tqdm import tqdm
    from torch.utils.data import DataLoader
    #ds = KITTIconverted('home/vision/project/instance_dataset_gen/Test_data/Kitti/npydata', mode='val')
    #dl = DataLoader(ds, batch_size = 4, shuffle = True, num_workers = 4, drop_last = True)

    home = False
    if home:
        datapath = '../../pointcloud_spherical_gen/Test_data/Kitti/npydata'
        listpath = '../../pointcloud_spherical_gen/Test_data/Kitti/data_list.txt'
    else:
        datapath = '../../instance_dataset_gen/Test_data/Kitti/npydata'
        listpath = '../../instance_dataset_gen/Test_data/Kitti/data_list.txt'    

    dl = get_data_loader(datapath, 5, listpath=listpath)
    for sample in dl:
        print("image size from dataloader: ", sample['img'].shape)
