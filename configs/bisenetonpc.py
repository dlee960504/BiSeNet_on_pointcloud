## bisenet_on_pc
cfg = dict(
    model_type='bisenetonpc',
    num_cls=28,
    num_aux_heads=4,
    lr_start = 5e-2,
    weight_decay=5e-4,
    warmup_iters = 1000,
    max_iter = 150000,
    im_root='../../pointcloud_spherical_gen/Test_data/Kitti/npydata',
    train_im_anns='../../pointcloud_spherical_gen/Test_data/Kitti/data_list.txt',
    val_im_anns='../../pointcloud_spherical_gen/Test_data/Kitti/data_list.txt',
    scales=[0.25, 2.],
    cropsize=[512, 1024],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)