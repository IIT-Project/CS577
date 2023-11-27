_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/animalpose.py'
]
evaluation = dict(interval=10, metric='mAP', save_best='AP')
checkpoint_config = dict(interval=10)

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    dataset_joints=20,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ])

data_cfg = dict(
    image_size=256,
    base_size=256,
    base_sigma=2,
    heatmap_size=[32],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='AssociativeEmbedding',
    pretrained='D:\\工作和学习文档\\IIT\\CS577\\project\\vitpose\\mae_pretrain_vit_base.pth',
    backbone=dict(
            type='ViT',
            img_size=(256, 256),
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ratio=2,
            use_checkpoint=True,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.2,
        ),
    keypoint_head=dict(
        type='AESimpleHead',
        in_channels=768,
        num_joints=20,
        num_deconv_layers=0,
        tag_per_joint=True,
        with_ae_loss=[True],
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=20,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0])),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True],
        with_ae=[True],
        project2image=True,
        align_corners=False,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=True))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
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
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline

data_root = 'D:\\工作和学习文档\\IIT\\CS577\\project\\ViTPose-main\\data\\pet'
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=16),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpAnimalDataset',
        ann_file=f'{data_root}/annotations/animalpose_train.json',
        img_prefix=f'{data_root}\\VOC2012\\JPEGImages\\',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpAnimalDataset',
        ann_file=f'{data_root}/annotations/animalpose_val.json',
        img_prefix=f'{data_root}\\VOC2012\\JPEGImages\\',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpAnimalDataset',
        ann_file=f'{data_root}/annotations/animalpose_test.json',
        img_prefix=f'{data_root}\\VOC2012\\JPEGImages\\',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}))