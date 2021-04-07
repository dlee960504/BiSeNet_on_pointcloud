## @file bisenet_on_pc2.py
home = False;
if home:
    im_root='../../pointcloud_spherical_gen/Test_data/Kitti/npydata'
    val_im_root='../../pointcloud_spherical_gen/Test_data/Kitti/npydata'
    train_im_anns='../../pointcloud_spherical_gen/Test_data/Kitti/data_list.txt'
    val_im_anns='../../pointcloud_spherical_gen/Test_data/Kitti/data_list.txt'
else:
    im_root='../../BiSeNet/datasets/Kitti/npydata'
    val_im_root='../../BiSeNet/datasets/Kitti_test/npydata'
    train_im_anns='../../BiSeNet/datasets/Kitti/data_list.txt'
    val_im_anns='../../BiSeNet/datasets/Kitti_test/data_list.txt'

cfg = dict(
    model_type='bisenetonpc2',
    num_cls=4,
    num_aux_heads=4,
    lr_start = 5e-2,
    weight_decay=5e-4,
    warmup_iters = 1000,
    max_iter = 150000,
    max_epoch = 201,
    im_root=im_root,
    val_im_root=val_im_root,
    train_im_anns=train_im_anns,
    val_im_anns=val_im_anns,
    scales=[0.25, 2.],
    cropsize=[512, 1024],
    batch_size=4,
    use_fp16=True,
    use_sync_bn=False,
    respth='../res',
)

# RGB
color_code = [[255, 255, 255], [0, 0, 255], [255, 0, 0], [0, 255, 0]]