# from pathlib import Path
# import mmtrack
# mmtrack_path = Path(mmtrack.__file__).parent.parent
# # Choose to use a config and initialize the detector
# config = mmtrack_path / '/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'

config =  '/home/appuser/mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
_base_ = [str(config)]
# _base_ = ['./bytetrack_yolox_x_crowdhuman_mot17-private-half.py']

img_scale = (896, 1600)

model = dict(
    detector=dict(input_size=img_scale, random_size_range=(20, 36)),
    tracker=dict(
        weight_iou_with_det_scores=False,
        match_iou_thrs=dict(high=0.3),
    ))

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        dataset=dict(
            ann_file=[
                "../mnt/pole/data-ntis/projects/korpusy_cv/pigleg_surgery/detection/test/pigleg_cocovid.json",
                # 'data/MOT20/annotations/train_cocoformat.json',
                # 'data/crowdhuman/annotations/crowdhuman_train.json',
                # 'data/crowdhuman/annotations/crowdhuman_val.json'
            ],
            img_prefix=[
                # 'data/MOT20/train', 'data/crowdhuman/train',
                # 'data/crowdhuman/val'
                "../mnt/pole/data-ntis/projects/korpusy_cv/pigleg_surgery/detection/"
            ]),
        pipeline=train_pipeline),
    val=dict(
        ann_file = "../mnt/pole/data-ntis/projects/korpusy_cv/pigleg_surgery/detection/test/pigleg_cocovid.json",
        img_prefix= "../mnt/pole/data-ntis/projects/korpusy_cv/pigleg_surgery/detection/",
pipeline=test_pipeline),
    test=dict(
        ann_file = "../mnt/pole/data-ntis/projects/korpusy_cv/pigleg_surgery/detection/test/pigleg_cocovid.json",
        img_prefix= "../mnt/pole/data-ntis/projects/korpusy_cv/pigleg_surgery/detection/",
        pipeline=test_pipeline))

checkpoint_config = dict(interval=1)
evaluation = dict(metric=['bbox', 'track'], interval=1)