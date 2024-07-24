default_scope = 'mmpretrain'

interval = 1

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=10
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

custom_hooks = [
    dict(
        type='SetDropoutRateHook',
        drop_img_rate=0,
        drop_token_rate=0,
        drop_extra_rate=0,
        drop_modality_rate=0,
    ),
    # dict(
    #     type='SetBalancedDatasetHook',
    # )
]
log_processor = dict(by_epoch=True)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
# work_dir = '../work_dirs/shengtaijishen/cn_clip_1img_huge_20240315'



vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='WandbVisBackend', init_kwargs=dict(project='market', group='cnclip', name='cn_clip_5img_v3_notefeat_nobalance')),
]

visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False
num_classes = 37

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    to_rgb=True,
    pad_value=0,
)

img_size = (224, 224)
num_imgs = 1

# hf_cfg_model_name = 'OFA-Sys/chinese-clip-vit-large-patch14-336px'
# hf_cfg_model_name = 'OFA-Sys/chinese-clip-vit-large-patch14'
hf_cfg_model_name = '/mnt/nlp-ali/usr/yanshilin/CKPTS/chinese-clip-vit-huge-patch14'

# init_cfg = dict(
#     type='Pretrained',
#     checkpoint='work_dirs/market/v3_pretrain_400000_fp.pth',
# )
init_cfg = None

# model settings
model = dict(
    init_cfg=init_cfg,
    type='NoteCLIPAllFeatFusionClassifier',
    freeze_backbone=False,
    data_preprocessor=data_preprocessor,
    is_pooling_feats=False,
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
        vision_project=dict(type='Linear', in_features=1280, out_features=1024),
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
)

randomness = dict(seed=None, deterministic=False)

load_train_pipeline = [
    dict(type='LoadImageFromUrl', to_float32=True, ignore_empty=True, mean_rgb=data_preprocessor['mean']),
    dict(type='Resize', scale=img_size, interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
]

post_pipeline = [
    dict(type='PackInputs', algorithm_keys=[
        'input_main_string',
        # 'input_user_string',
        # 'input_note_string',
        # 'input_main_int',
        # 'input_user_int',
        # 'input_note_int',
        # 'input_main_float',
        # 'input_user_float',
        # 'input_note_float',
        'type',
        'img_attn_masks',
        'note_id',
        'urls']
         ),
]

load_test_pipeline = [
    dict(type='LoadImageFromUrl', to_float32=True, ignore_empty=True, mean_rgb=data_preprocessor['mean']),
    dict(type='Resize', scale=img_size, interpolation='bicubic'),
]

batch_size_per_gpu = 32
num_workers = 16
persistent_workers = True
dataset_type = 'ShengTaiShenheCLIPAllFeatMultiLabelDataset'
train_data_root = '/mnt/nlp-ali/usr/yanshilin/Projects/ecology_project/data/weile_mid_exp_multi_label_model_train_data2024_02'
val_data_root = '/mnt/nlp-ali/usr/yanshilin/Projects/ecology_project/data/weile_mid_exp_multi_label_model_train_data2024_02'

indices = None
# indices = list(range(0, 512))
metainfo = dict(classes=['通过', '工业化', '未成年人不当行为', '搬运', '未成年人不推荐', '轻微不适', '男性不宜', '虚构内容', 
    '导流', '舆情热点跟风', '风险社交', '包含医疗建议', '特殊行业营销', '风险交互', '违纪违规', '风险信息', '无信息量', 'LGBT', 
    '低俗', '重度不适', '平台保护', '套路营销', '诱导互动', '虚假不实', '土味内容', '剧情演绎', '猎奇重口', '无资质科普', 
    '未成年不良价值观', '医学科普', '低质创作', '谩骂攻击', '危险行为', '标题党', '风险推广', '同质化内容', '音画低质'])
# metainfo = dict(classes=['无', '优', '差'])

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
        indices=indices,
        main_feat_cfg=dict(
            redkv_prefix='sly_shengtaijishen',
            redkv_conn=dict(
                host='10.99.232.171',
                port='12345',
                db=0
            ),
            max_text_length=512,
            tokenizer=hf_cfg_model_name,
        ),
        data_root=train_data_root,
        ann_file='weile_20240315_train.txt',
        load_pipeline=load_train_pipeline,
        pipeline=post_pipeline
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
        indices=indices,
        main_feat_cfg=dict(
            redkv_prefix='sly_shengtaijishen',
            redkv_conn=dict(
                host='10.99.232.171',
                port='12345',
                db=0
            ),
            max_text_length=512,
            tokenizer=hf_cfg_model_name,
        ),
        data_root=val_data_root,
        ann_file='weile_20240315_test.txt',
        load_pipeline=load_test_pipeline,
        pipeline=post_pipeline,
        test_mode=True
    )
)

val_evaluator = [
    dict(
        type='ShengTaiShenheMultiLabelFixedPMetrics',
        num_classes=num_classes,
        fixed_precision=0.9,
        class_names=metainfo['classes']
    ),
    dict(
        type='ShengTaiShenheMultiLabelFixedPMetrics',
        num_classes=num_classes,
        fixed_precision=0.95,
        class_names=metainfo['classes']
    )
]
test_dataloader = val_dataloader
test_evaluator = val_evaluator

base_lr = 1e-4
max_epochs = 10
# max_iters = 4000 * 100
# max_iters = 4000

# # compile = True
# # find_unused_parameters = True
# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     dtype='float16',
#     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05)
# )

strategy = dict(
    type='DeepSpeedStrategy',
    fp16=dict(
        enabled=True,
        auto_cast=False,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    # gradient_clipping=0.1,
    inputs_to_half=['inputs'],
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        allgather_bucket_size=2e8,
        reduce_scatter=True,
        reduce_bucket_size='auto',
        overlap_comm=True,
        contiguous_gradients=True,
    ),
)

optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05
    ),
)

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