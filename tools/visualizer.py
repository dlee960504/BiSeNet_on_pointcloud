import sys
sys.append('.')
import numpy as np
import pcl

def visualize_seg(pred_mask, color_map):
    assert(len(pred_mask.shape) == 2)
    num_cls = len(color_map)

    out = np.zeross((pred_mask.shape[0], pred_mask.shape[1], 3))

    for i in range(num_cls):
        out[pred_mask==i, :] = color_map[i]

    return out

def back_project(pred_mask, color_map):
    lidar = np.load()

    pred_mask_raw = pred_mask.reshape(-1,1)

