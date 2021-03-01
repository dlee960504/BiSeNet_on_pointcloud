import sys
sys.append('.')
import numpy as np
import pcl
from configs import color_map

cmap = color_map['bisenetonpc'] 

def visualize_seg(pred_mask):
    assert(len(pred_mask.shape) == 2)
    num_cls = len(cmap)

    out = np.zeross((pred_mask.shape[0], pred_mask.shape[1], 3))

    for i in range(num_cls):
        out[pred_mask==i, :] = cmap[i]

    return out

def back_project(pred_mask, npydata):
    lidar = np.load(npydata).astype(np.float32, copy=False)[ :, :, :5]

    pred_mask_raw = pred_mask.reshape(-1,1)

