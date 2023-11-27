_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/coco.py'
]
checkpoint_config = dict(interval=50)
evaluation = dict(interval=50, metric='mAP', save_best='AP')

optimizer = dict(type='AdamW', lr=0.0015, betas=(0.9, 0.999), weight_decay=0.1,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                                    num_layers=12, 
                                    layer_decay_rate=0.9,
                                    custom_keys={
                                            'bias': dict(decay_multi=0.),
                                            'pos_embed': dict(decay_mult=0.),
                                            'relative_position_bias_table': dict(decay_mult=0.),
                                            'norm': dict(decay_mult=0.)
                                            }
                                    )
                )

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[200, 260])
total_epochs = 300
channel_cfg = dict(
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

data_cfg = dict(
    image_size=512,
    base_size=256,
    base_sigma=2,
    heatmap_size=[128, 256],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=2,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='AssociativeEmbedding',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(512, 512),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=2,
        use_checkpoint=True,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        drop_path_rate=0.2,
        out_indices=[11],
        stages=4,
        last_norm=True,
    ),
    keypoint_head=dict(
        type='AEHigherResolutionHead',
        in_channels=768,
        num_joints=17,
        num_deconv_layers=1,
        num_deconv_filters=[128,],
        num_deconv_kernels=[4,],
        num_deconv_layers_light=1,
        num_deconv_filters_light=(256,),
        num_deconv_kernels_light=(4,),
        num_basic_blocks=4,
        cat_output=[True],
        tag_per_joint=True,
        with_ae_loss=[True, False],
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0])),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True, True],
        with_ae=[True, False],
        project2image=False,
        align_corners=True,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=True,
        use_udp=True))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40,
        use_udp=True),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=30,
        use_udp=True,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1], use_udp=True),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ],
        use_udp=True,
        ),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/coco'
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=12),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
