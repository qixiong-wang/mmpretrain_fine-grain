_base_ = '../_base_/default_runtime.py'

interval = 1

log_processor = dict(by_epoch=True)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)



vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='WandbVisBackend', init_kwargs=dict(project='market', group='cnclip', name='cn_clip_5img_v3_notefeat_nobalance')),
]

visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False
num_classes = 23

data_preprocessor = dict(
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

img_size = (224, 224)
num_imgs = 1

hf_cfg_model_name = 'openai/clip-vit-large-patch14',

# init_cfg = dict(
#     type='Pretrained',
#     checkpoint='work_dirs/market/v3_pretrain_400000_fp.pth',
# )
init_cfg = None

# model settings
model = dict(
    init_cfg=init_cfg,
    type='CLIPFusionClassifier',
    freeze_backbone=True,
    data_preprocessor=data_preprocessor,
    vision_project=dict(type='Linear', in_features=1024, out_features=1024),
    text_project=dict(type='Linear', in_features=768, out_features=1024),
    backbone=dict(
        type='HFCNCLIPVision',
        from_pretrained=True,
        config_dir=hf_cfg_model_name,
        lora_cfg=dict(
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,

        ),
    ),
    text_backbone=dict(
        type='HFCNCLIPText',
        from_pretrained=True,
        config_dir=hf_cfg_model_name,
        lora_cfg=dict(
            r=32,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.0,
        ),
    ),
    neck=dict(
        type='TransformerFusionNeck',
        embed_dims=1024,
        num_modality=2,
        with_cls_token=True,
        num_encoder_layers=3,
    ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=num_classes,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
    ),
    # tokenizer=dict(
    #     type='AutoTokenizer',
    #     name_or_path='openai/clip-vit-base-patch16',
    #     use_fast=False),
    # vocab_size=49408,
    # transformer_width=512,
    # proj_dim=512,
)


randomness = dict(seed=None, deterministic=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_size, interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs', algorithm_keys=['input_main_string']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_size, interpolation='bicubic'),
    dict(type='PackInputs', algorithm_keys=['input_main_string']),
]

batch_size_per_gpu = 8
num_workers = 8
persistent_workers = False
dataset_type = 'IMDBDataset'
train_data_root = './'
val_data_root = './'

metainfo = dict(classes=['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror', 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music', 'Animation', 'Musical', 'Western', 'Sport', 'Short', 'Film-Noir'])


train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        num_imgs=num_imgs,
        feat_cfg=dict(
            max_text_length=512,
            tokenizer=hf_cfg_model_name,
        ),
        data_root=train_data_root,
        ann_file='mmimdb/split.json',
        split = 'train',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu*2,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        num_imgs=num_imgs,
        feat_cfg=dict(
            max_text_length=512,
            tokenizer=hf_cfg_model_name,
        ),
        data_root=val_data_root,
        ann_file='mmimdb/split.json',
        split = 'test',
        pipeline=test_pipeline,
        test_mode=True
    )
)

val_evaluator = [
    dict(
        type='MultiLabelMetric',
    ),
]
test_dataloader = val_dataloader
test_evaluator = val_evaluator

base_lr = 1e-5
max_epochs = 10
# max_iters = 4000 * 100
# max_iters = 4000

# # compile = True
# # find_unused_parameters = True
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05)
)

# strategy = dict(
#     type='DeepSpeedStrategy',
#     fp16=dict(
#         enabled=True,
#         auto_cast=False,
#         fp16_master_weights_and_grads=False,
#         loss_scale=0,
#         loss_scale_window=500,
#         hysteresis=2,
#         min_loss_scale=1,
#         initial_scale_power=15,
#     ),
#     # gradient_clipping=0.1,
#     inputs_to_half=['inputs'],
#     zero_optimization=dict(
#         stage=2,
#         allgather_partitions=True,
#         allgather_bucket_size=2e8,
#         reduce_scatter=True,
#         reduce_bucket_size='auto',
#         overlap_comm=True,
#         contiguous_gradients=True,
#     ),
# )

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05
#     ),
# )

runner_type = 'FlexibleRunner'
# runner_type = 'Runner'


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=base_lr / 100,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr*0.01,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()