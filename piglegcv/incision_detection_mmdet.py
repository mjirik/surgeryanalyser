# Some basic setup

# import some common libraries
import logging

import cv2
import mmcv.utils

# logger = mmcv.utils.get_logger(name=__file__, log_level=logging.DEBUG)
from loguru import logger

import matplotlib.pyplot as plt
import numpy as np

# Check Pytorch installation
import torch
import torchvision

logger.debug(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet

logger.debug(f"mmdet.version={mmdet.__version__}")

# Check mmcv installation
from mmcv.ops import get_compiler_version, get_compiling_cuda_version

logger.debug(get_compiling_cuda_version())
logger.debug(get_compiler_version())
import os.path as osp
from pathlib import Path
from pprint import pformat, pprint
from typing import Optional

from mmcv import Config
from mmdet.apis import (
    inference_detector,
    init_detector,
    set_random_seed,
    show_result_pyplot,
    train_detector,
)
from mmdet.datasets import build_dataset

mmdetection_path = Path(mmdet.__file__).parent.parent

import json
import os
from pathlib import Path
from typing import Union

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from tools import load_json, save_json

scratchdir = Path(os.getenv("SCRATCHDIR", "."))
logname = Path(os.getenv("LOGNAME", "."))

# from loguru import logger

local_input_data_dir = Path(scratchdir) / "data/orig/"
local_output_data_dir = Path(scratchdir) / "data/processed/"


def prepare_cfg(
    local_input_data_dir: Path,
    local_output_data_dir: Path,
    checkpoint_pth: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    skip_data=False,
):
    if checkpoint_pth == None:
        checkpoint_pth = (
            scratchdir
            / "checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth"
        )
    checkpoint_pth = str(checkpoint_pth)

    work_dir = (
        str(work_dir) if work_dir else str(local_output_data_dir / "tutorial_exps")
    )

    logger.debug(f"outputdir={local_output_data_dir}")
    logger.debug(f"input_data_dir={local_input_data_dir}")
    logger.debug(f"input_data_dir exists={local_input_data_dir.exists()}")
    logger.debug(f'input_data_dir glob={str(list(local_input_data_dir.glob("**/*")))}')

    # # Choose to use a config and initialize the detector
    # config = mmdetection_path / 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
    # logger.debug(f"config.exists={config.exists()}")
    # # Setup a checkpoint file to load
    # checkpoint_pth = scratchdir / 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
    # logger.debug(f"checkpoint_pth.exists={checkpoint_pth.exists()}")
    #
    # # Set the device to be used for evaluation
    # device='cuda:0'
    #
    # # Load the config
    # config = mmcv.Config.fromfile(config)
    # # Set pretrained to be None since we do not need pretrained model here
    # config.model.pretrained = None
    #
    # # Initialize the detector
    # model = build_detector(config.model)
    #
    # # Load checkpoint
    # checkpoint = load_checkpoint(model, str(checkpoint_pth), map_location=device)
    #
    # # Set the classes of models for inference
    # model.CLASSES = checkpoint['meta']['CLASSES']
    #
    # # We need to set the model's cfg for inference
    # model.cfg = config
    #
    # # Convert the model to GPU
    # model.to(device)
    # # Convert the model into evaluation mode
    # model.eval()
    #
    # # Use the detector to do inference
    # img = mmdetection_path / 'demo/demo.jpg'
    # result = inference_detector(model, img)
    # model.show_result(img, result, out_file=local_output_data_dir / 'demo_output.jpg')# save image with result

    # My dataset training
    cfg = Config.fromfile(
        mmdetection_path
        / "configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py"
    )

    cfg.dataset_type = "CocoDataset"
    cfg.data_root = str(local_input_data_dir)
    cfg.classes = ("incision",)
    if not skip_data:
        # Modify dataset type and path

        cfg.data.test.type = "CocoDataset"
        cfg.data.test.data_root = str(local_input_data_dir)
        cfg.data.test.ann_file = "annotations/instances_default.json"
        cfg.data.test.img_prefix = "images/"
        cfg.data.test.classes = cfg.classes

        cfg.data.train.type = "CocoDataset"
        cfg.data.train.data_root = str(local_input_data_dir)
        cfg.data.train.ann_file = "annotations/instances_default.json"
        cfg.data.train.img_prefix = "images/"
        cfg.data.train.classes = cfg.classes

        cfg.data.val.type = "CocoDataset"
        cfg.data.val.data_root = str(local_input_data_dir)
        cfg.data.val.ann_file = "annotations/instances_default.json"
        cfg.data.val.img_prefix = "images/"
        cfg.data.val.classes = cfg.classes

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 1
    # If we need to finetune a model based on a pre-trained detector, we need to
    # use load_from to set the path of checkpoints.
    cfg.load_from = checkpoint_pth

    # Set up working dir to save files and logs.
    cfg.work_dir = work_dir

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    # cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 12
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ]

    # We can initialize the logger for training and have a look
    # at the final config used for training
    # print(f'Config:\n{cfg.pretty_text}') # does not work for paths beginning '/' because of bug in lib2to3

    logger.debug(f"cfg=\n{pformat(cfg)}")

    return cfg


def train(cfg):
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    logger.debug(f"classes={datasets[0].CLASSES}")

    # Build the detector
    model = build_detector(cfg.model)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    train_detector(model, datasets, cfg, distributed=False, validate=True)
    return model


def run_incision_detection(
    img,
    local_output_data_dir: Optional[Path] = None,
    meta: Optional[dict] = None,
    expected_incision_size_mm=70,
    device="cuda",
):
    # todo J. Viktora
    # nemělo by tady být spíš device="cuda" ? To ale nefunguje protože: RuntimeError: nms_impl: implementation for device cuda:0 not found.
    # img = mmcv.imread(str(img_fn))

    if meta is None:
        meta = {}
    checkpoint_path = (
        Path(__file__).parent
        / "resources/incision_detection_models/220326_234659_mmdet.pth"
    )
    logger.debug(f"checkpoint_path.exists={checkpoint_path.exists()}")
    logger.debug(f"img.shape={img.shape}, max(img)={np.max(img)}")
    # logger.debug(f"img_fn={img_fn}")

    # img_fn = Path(img_fn)
    if local_output_data_dir is not None:
        local_output_data_dir = Path(local_output_data_dir)

    # My dataset training
    cfg = Config.fromfile(
        mmdetection_path
        / "configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py"
    )

    cfg.dataset_type = "CocoDataset"
    # cfg.data_root = str(local_input_data_dir)
    cfg.classes = ("incision",)
    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 1

    model = init_detector(
        cfg,
        str(checkpoint_path),
        device=device
        # device='cuda:0'
    )
    # logger.debug(f"cfg=\n{pformat(cfg)}")
    result = inference_detector(model, img)
    if local_output_data_dir is not None:
        model.show_result(
            img, result, out_file=local_output_data_dir / f"incision_full.jpg"
        )  # save image with result

    # get cropped incision
    class_id = 0
    # obj_in_class_id = 0
    bboxes = result[class_id]
    bbox_sizes, imgs = save_incision_cropped_images(img, local_output_data_dir, bboxes)
    # predict_image_with_cfg(cfg, model, img_fn, local_output_data_dir)
    if len(bbox_sizes) > 0:
        bbox_sizes = np.asarray(bbox_sizes)
        incision_size_px = np.median(bbox_sizes, axis=0)  #  first is the smaller size
        pixelsize_m = 0.001 * expected_incision_size_mm / incision_size_px[1]
    else:
        pixelsize_m = None
    # json_file = Path(local_output_data_dir) / "meta.json"
    meta.update(
        {
            "pixelsize_m_by_incision_size": pixelsize_m,
            "incision_bboxes": bboxes.tolist(),
        }
    )

    return imgs, bboxes


def save_incision_cropped_images(img, local_output_data_dir, bboxes, suffix=""):
    logger.debug(f"number of detected incisions = {len(bboxes)}")
    imgs = []
    bbox_sizes = []  # used for resolution evaluation
    for i, bbox in enumerate(bboxes):

        imcr = img[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]

        sz = sorted([int(bbox[3]) - int(bbox[1]), int(bbox[2]) - int(bbox[0])])
        bbox_sizes.append(sz)
        if local_output_data_dir is not None:
            cv2.imwrite(str(local_output_data_dir / f"incision_crop_{i}{suffix}.jpg"), imcr)
        imgs.append(imcr)
        # plt.imshow(imcr[:, :, ::-1])
    return bbox_sizes, imgs


def predict_image_with_cfg(cfg, model, img_fn, local_output_data_dir):
    # img_fn = local_input_data_dir / '/images/10.jpg'
    img = mmcv.imread(img_fn)

    model.cfg = cfg
    result = inference_detector(model, img)
    # show_result_pyplot(model, img, result)
    model.show_result(
        img, result, out_file=local_output_data_dir / f"output_{img_fn.stem}.jpg"
    )  # save image with result
    return result


def predict_images(cfg, model, local_input_data_dir: Path, local_output_data_dir):

    filelist = []
    if local_input_data_dir.is_dir():
        filelist = list(local_input_data_dir.glob("*.jpg"))
        filelist.extend(list(local_input_data_dir.glob("*.png")))
    else:
        filelist = [local_input_data_dir]

    results = []
    for img_fn in filelist:
        result = predict_image_with_cfg(cfg, model, img_fn, local_output_data_dir)
        results.append(result)

    # # print all files in input dir recursively to check everything
    logger.debug(str(list(Path(local_output_data_dir).glob("**/*"))))
    return results


if __name__ == "__main__":
    cfg = prepare_cfg(local_input_data_dir, local_output_data_dir)
    predict_images(local_input_data_dir / "images")
