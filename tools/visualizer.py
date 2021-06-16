import sys, os

from torch import convolution
sys.path.append('.')
import numpy as np
import pcl
import pcl.pcl_visualization
from plyfile import PlyData, PlyElement
from configs import color_map
import datetime

cmap = color_map['bisenetonpc']

def ply2npy(file_dir):
    plydata = PlyData.read(file_dir)
    x = np.array(plydata['vertex'].data['x'])
    y = np.array(plydata['vertex'].data['y'])
    z = np.array(plydata['vertex'].data['z'])

    xyz = np.vstack([x, y, z]).transpose()

    r = np.array(plydata['vertex'].data['red'])
    g = np.array(plydata['vertex'].data['green'])
    b = np.array(plydata['vertex'].data['blue'])

    # mean rgb to get grayscale
    rgb_mean = np.mean(np.array([r, g, b]), axis=0).astype('uint8')

    # grayscale according to ITU-R BT.601
    gray = 0.299*r + 0.587*g + 0.114*b
    gray = gray.astype('uint8')

    #print("xyz shape: {}".format(xyz.shape))


def colorize_3c(preds):

    preds = np.squeeze(preds)
    num_cls = len(cmap)

    out = np.zeros((preds.shape[0], preds.shape[1], 3),dtype=int)

    for i in range(num_cls):
        out[preds==i, :] = cmap[i]

    return out

def colorize(preds):
    cmap_1c = []
    for cc in cmap:
        temp = cc[0]<<16 | cc[1]<<8 | cc[0]<<0
        cmap_1c.append(temp)

    preds = np.squeeze(preds)
    num_cls = len(cmap_1c)

    out = np.zeros((preds.shape[0], preds.shape[1], 1),dtype=int)

    for i in range(num_cls):
        out[preds==i, :] = cmap_1c[i]

    return out

def back_project(xyz, preds, pop_up=False):
    assert preds.shape[2] == 1 and xyz.shape[2] == 3, 'check if XYZRGB'

    rgb = np.zeros((preds.shape[0], preds.shape[1]), dtype=int)
    points = np.concatenate((xyz, preds), axis=2)
    points = np.reshape(points, (points.shape[0]*points.shape[1], -1), order='C')
    points = points.astype('float32')

    # declare point cloud object
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_array(points)

    if pop_up:
        # visualize
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorCloud(cloud)

        v = True
        while v:
            v = not(visual.WasStopped())
    
    save_dir = '../test/pc/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    pcl.save(cloud, save_dir + 'segmented_cloud.pcd')    
