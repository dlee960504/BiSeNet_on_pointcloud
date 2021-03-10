import sys, os
sys.path.append('..')
import numpy as np
from PIL import Image
import pcl
import pyrealsense2 as rs
from plyfile import PlyData, PlyElement
from configs import color_map
import datetime

cmap = color_map['bisenetonpc']

# stream pointcloud from L515 and dump into .ply file
def streamL515(save_path='../datasets/rs_stream'):

    start_time = datetime.datetime.now()
    # create directory to save the stream
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    try:
        # context object
        pipe = rs.pipeline()

        # configure streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # prepare pointcloud object
        pc = rs.pointcloud()
        points = rs.points()

        # start streaming
        pipe.start(config)

        # colorizer from depth
        colorizer = rs.colorizer()

        # get frames
        go = True
        while go:
            curr_time = datetime.datetime.now()
            if (curr_time - start_time).total_seconds() > 60:
                break

            frames = pipe.wait_for_frames()
            colorized = colorizer.process(frames)

            # create save_to_ply object
            ply = rs.save_to_ply(os.path.join(save_path, 'raw_data.ply'))
            ply.set_option(rs.save_to_ply.option_ply_binary, True)
            ply.set_option(rs.save_to_ply.option_ply_normals, False)
            ply.set_option(rs.save_to_ply.option_ply_mesh, False)
            ply.process(colorized)            

    except Exception as e:
        print(e)
        pass
    finally:
        pipe.stop()

def streamL515_IR(save_path='../datasets/rs_stream'):
    start_time = datetime.datetime.now()
    # create directory to save the stream
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    try:
        # context object
        pipe = rs.pipeline()

        # configure streams
        config = rs.config()
        config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

        # start streaming
        pipe.start(config)

        # get frames
        go = True
        while go:
            curr_time = datetime.datetime.now()
            if (curr_time - start_time).total_seconds() > 60:
                break

            frames = pipe.wait_for_frames()
            ir = frames.first(rs.stream.infrared)
            if not ir:
                print('frame failure')
                continue
            np_img = np.asanyarray(infrared_frame.get_data())
            np.save(os.path.join(save_path, 'raw_IR'), np_img)
        
        pipe.stop()
          

    except Exception as e:
        print(e)
        pass
    
    finally:
        print()
        #pipe.stop()


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

if __name__ == '__main__':
    streamL515_IR()


