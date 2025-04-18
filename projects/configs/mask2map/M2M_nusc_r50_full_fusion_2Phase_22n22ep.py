_base_ = [
    "../datasets/custom_nus-3d.py",
    "../_base_/default_runtime.py"]

work_dir = None
# resume_from = "./work_dirs/M2M_nusc_r50_full_fusion_2Phase_22n22ep/epoch_12.pth"
load_from = "./work_dirs/M2M_nusc_r50_full_fusion_1Phase_22n22ep/latest.pth"
resume_optimizer = False

#
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
voxel_size = [0.15, 0.15, 20.0]
dbound = [1.0, 35.0, 0.5]

lidar_point_cloud_range = [-15.0, -30.0, -5.0, 15.0, 30.0, 3.0]
lidar_voxel_size = [0.1, 0.1, 0.2]

grid_config = {
    "x": [-30.0, -30.0, 0.15],  # useless
    "y": [-15.0, -15.0, 0.15],  # useless
    "z": [-10, 10, 20],         # useless
    "depth": [1.0, 35.0, 0.5],  # useful
}


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
# map has classes: divider, ped_crossing, boundary
map_classes = ["divider", "ped_crossing", "boundary"]
num_vec = 50
fixed_ptsnum_per_gt_line = 20  # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 200 #60m, 0.3m
bev_w_ = 100 #30m, 0.3m
queue_length = 1  # each sequence contains `queue_length` frames.

dn_enabled = True

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=True,
    seg_classes=3, # test
    feat_down_sample=32,
    pv_thickness=1,
)

with_cp_backbone = True
with_cp_pts_decoder = True

modality='fusion'

model = dict(
    type="Mask2Map",
    use_grid_mask=True,
    video_test_mode=False,
    modality=modality,
    lidar_encoder=dict(
        voxelize=dict(
            max_num_points=10,
            point_cloud_range=lidar_point_cloud_range,
            voxel_size=lidar_voxel_size,
            max_voxels=[90000, 120000]),
        backbone=dict(
            type='SparseEncoder',
            in_channels=5,
            sparse_shape=[300, 600, 41],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                        128)),
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
            block_type='basicblock'
        ),
    ),
    pretrained=dict(img="ckpts/resnet50-19c8e357.pth"),
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
    ),
    
    img_neck=dict(
        type="FPN",
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=_num_levels_,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type="Mask2MapHead_2Phase",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_vec_one2one=num_vec,
        num_pts_per_vec=fixed_ptsnum_per_pred_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        dn_enabled=dn_enabled,
        dn_weight=1,
        cls_join=True,
        tr_cls_join=False,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=aux_seg_cfg,
        transformer=dict(
            type="Mask2Map_Transformer_2Phase_CP",
            modality=modality,
            fuser=dict(
                type='ConvFuser',
                in_channels=[_dim_, 256],
                out_channels=_dim_,
            ),
            rotate_prev_bev=True,
            dropout=0.1,  # cross attention dropout
            thr=0.8,  # point threshold
            dn_enabled=dn_enabled,
            dn_group_num=5,
            dn_noise_scale=0.01,
            thresh_of_mask_for_pos=0.3,
            mask_noise_scale=0.1,
            pts2mask_noise_scale=0.1,
            num_vec_one2one=num_vec,
            embed_dims=_dim_,
            bev_encoder=None,
            bev_neck=dict(
                type="MSDeformAttnPixelDecoder",
                num_outs=4,
                in_channels=[256, 256, 256, 256],
                strides=[1, 2, 4, 8],
                norm_cfg=dict(type="GN", num_groups=32),
                act_cfg=dict(type="ReLU"),
                encoder=dict(  # DeformableDetrTransformerEncoder
                    num_layers=6,
                    layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                        self_attn_cfg=dict(
                            embed_dims=256,
                            num_heads=8,
                            num_levels=3,
                            num_points=4,
                            dropout=0.0,
                            batch_first=True,
                        ),  # MultiScaleDeformableAttention
                        ffn_cfg=dict(
                            embed_dims=256,
                            feedforward_channels=1024,
                            num_fcs=2,
                            ffn_drop=0.0,
                            act_cfg=dict(type="ReLU", inplace=True),
                        ),
                    ),
                ),
                positional_encoding=dict(num_feats=128, normalize=True),
            ),
            encoder=dict(
                type="LSSTransform",
                in_channels=_dim_,
                out_channels=_dim_,
                feat_down_sample=32,
                pc_range=point_cloud_range,
                voxel_size=voxel_size,
                dbound=dbound,
                downsample=2,
                loss_depth_weight=3.0,
                depthnet_cfg=dict(use_dcn=False, with_cp=False, aspp_mid_channels=96),
                grid_config=grid_config,
            ),
            segm_decoder=dict(
                type="Mask2FormerTransformerDecoder",
                return_intermediate=True,
                num_layers=3,
                layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                    self_attn_cfg=dict(
                        embed_dims=256, num_heads=8, dropout=0.0, batch_first=True
                    ),  # MultiheadAttention
                    cross_attn_cfg=dict(
                        embed_dims=256, num_heads=8, dropout=0.0, batch_first=True
                    ),  # MultiheadAttention
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            decoder=dict(
                type="MapTRDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DecoupledDetrTransformerDecoderLayer_CP",
                    with_cp=with_cp_pts_decoder,
                    num_vec=num_vec,
                    num_pts_per_vec=fixed_ptsnum_per_pred_line,
                    attn_cfgs=[
                        dict(type="MultiheadAttention", embed_dims=_dim_, num_heads=8, dropout=0.1),
                        dict(type="MultiheadAttention", embed_dims=_dim_, num_heads=8, dropout=0.1),
                        dict(type="CustomMSDeformableAttention", embed_dims=_dim_, num_levels=4),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="MapTRNMSFreeCoder",
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes,
        ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type="L1Loss", loss_weight=0.0),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
        loss_pts=dict(type="PtsL1Loss", loss_weight=5.0),
        loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.005),
        loss_seg=dict(type="SimpleLoss", pos_weight=4.0, loss_weight=1.0),
        loss_pv_seg=dict(type="SimpleLoss", pos_weight=1.0, loss_weight=2.0),
        loss_segm_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * len(map_classes) + [0.1],
        ),
        loss_segm_mask=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0
        ),
        loss_segm_dice=dict(
            type="DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="MapTRAssigner",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBoxL1Cost", weight=0.0, box_format="xywh"),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=0.0),
                pts_cost=dict(type="OrderedPtsL1Cost", weight=5),
                pc_range=point_cloud_range,
            ),
            assigner_segm=dict(
                type="MaskHungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=2.0),
                mask_cost=dict(type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
                dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
            ),
        )
    ),
)

dataset_type = "CustomNuScenesOfflineLocalMapDataset"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")

reduce_beams=32
load_dim=5
use_dim=5

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=9, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='CustomPointsRangeFilter', point_cloud_range=lidar_point_cloud_range),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='CustomPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='PadMultiViewImageDepth', size_divisor=32), 
    dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,class_names=map_classes),
    dict(type='CustomCollect3D', keys=['img', 'gt_depth', 'points'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=9, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='CustomPointsRangeFilter', point_cloud_range=lidar_point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D', 
                with_gt=False, 
                with_label=False,
                class_names=map_classes),
            dict(type='CustomCollect3D', keys=['img', 'points'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "nuscenes_map_infos_temporal_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        aux_seg=aux_seg_cfg,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        dn_enabled=dn_enabled,
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "nuscenes_map_infos_temporal_val.pkl",
        map_ann_file=data_root + "nuscenes_mask2map_anns_val.json",
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "nuscenes_map_infos_temporal_val.pkl",
        map_ann_file=data_root + "nuscenes_mask2map_anns_val.json",
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names,
        modality=input_modality,
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline, metric='chamfer',
                  save_best='NuscMap_chamfer/mAP', rule='greater')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(max_keep_ckpts=100, interval=1)
find_unused_parameters=False

