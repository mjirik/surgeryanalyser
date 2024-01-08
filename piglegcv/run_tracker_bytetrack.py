# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser
import sys
import json
import shlex

import mmcv
import logging
from typing import List
from pathlib import Path

from loguru import logger

from mmtrack.apis import inference_mot, init_model
import tools
import numpy as np
from typing import Optional, List, Tuple

from mmdet.datasets.pipelines import Compose


def add_tracking_results(tracking_results, result):
    if result != None:
        frame_tr = []
        for i, tr in enumerate(result["track_bboxes"]):
            if len(tr) > 0:
                frame_tr.append(tr[0].tolist()[1:] + [i])
        tracking_results["tracks"].append(frame_tr)


def make_hash_from_model(model_file: Path):
    import hashlib

    hash_md5 = hashlib.md5()
    with open(model_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_hash_from_video_and_models(video_file: Path, trackers_config_and_checkpoints: List, additional_hash=""):
    """
    Hash is composed of video-parameters and model weights.

    It is concatenated hash of model weights and, size of frame, FPS, frame count, and perceptual
    hash of the first frame of the video.
    """
    # make peceptual hash of first video frame
    imgs = mmcv.VideoReader(str(video_file))
    frame_cnt = imgs.frame_cnt
    first_frame = next(imgs)
    logger.debug(f"{first_frame.shape=}")
    phash = tools.phash_image(first_frame)
    hash_hex = phash + "_"
    hash_hex += "0x%0.4X" % imgs.width
    hash_hex += "0x%0.4X" % imgs.height
    hash_hex += "0x%0.4X" % frame_cnt
    hash_hex += float(imgs.fps).hex()
    del imgs

    for tracker_config, tracker_checkpoint in trackers_config_and_checkpoints:
            hash_hex += make_hash_from_model(tracker_checkpoint)

    hash_hex += additional_hash
    return hash_hex

def _compare_composed_hashes(hash1:str, hash2:str, phash_distance_threshold=0.05)->bool:
    """Compare two hashes. Return True if they are equal.
    The first part is preceptual hash of firs frame of the video. The maximum distance between hashes might be defined.
    """

    hash1split = hash1.split("_", maxsplit=1)
    hash2split = hash2.split("_", maxsplit=1)
    if len(hash1split) != 2:
        return False
    if len(hash2split) != 2:
        return False

    distance = tools.phash_distance(hash1split[0], hash2split[0])
    logger.debug(f"hash {distance=}")
    logger.debug(f"{hash1=}, {hash2=}")
    if distance > phash_distance_threshold:
        return False

    if hash1split[1] != hash2split[1]:
        return False

    return True


def _should_do_tracking_based_on_hash(
        trackers_config_and_checkpoints:List, filename:Path, output_file_path:Path, additional_hash:str=""
) -> (Tuple)[bool, str]:

    hash_hex = make_hash_from_video_and_models(filename, trackers_config_and_checkpoints,
                                               additional_hash=additional_hash)
    run_tracking = True
    if output_file_path.exists():
        try:
            data = json.load(open(output_file_path, "r"))
            if "hash" in data:
                stored_hash = data["hash"]
                if _compare_composed_hashes(hash_hex, stored_hash):
                    run_tracking = False
        except Exception as e:
            logger.debug(f"Cannot read {Path(output_file_path).name}. Exception: {e}")

    logger.debug(f"{run_tracking=}, {hash_hex=}")
    return run_tracking, hash_hex



def main_tracker_bytetrack(
    trackers_config_and_checkpoints: list,
    # config_file,
    filename,
    # checkpoint,
    output_file_path: Path,
    device=None,
    class_names=[],
    additional_hash="",
    force_tracker:bool=False,
):
    """Run tracking on a video.
    trackers: is list of tuples (config_file, checkpoint)
    additional_hash: is a string that will be added to the hash of the first frame
    """
    logger.debug("main_tracker")
    run_tracking, hash_hex = _should_do_tracking_based_on_hash(
        trackers_config_and_checkpoints, filename,
        output_file_path, additional_hash=additional_hash)

    models = []
    for tracker_config, tracker_checkpoint in trackers_config_and_checkpoints:
        # build the model from a config file and a checkpoint file
        model = init_model(tracker_config, str(tracker_checkpoint), device=device)
        models.append(model)



    imgs = mmcv.VideoReader(str(filename))
    frame_cnt = imgs.frame_cnt

    if run_tracking or force_tracker:
        progress = tools.ProgressPrinter(frame_cnt)
        tracking_results = {"tracks": [None]*int(frame_cnt), "hash": hash_hex, "class_names": class_names}
        prev_track_ids = 20 * [-1]
        for frame_id, img in enumerate(imgs):
            frame_tr = []
            if not (frame_id % 50):
                
                logger.debug(
                    f"Tracking on frame {frame_id}, {progress.get_progress_string(float(frame_id))}"
                )
            for j, tracker in enumerate(models):

                result = inference_mot(
                    tracker, img, frame_id=frame_id
                )
 
                if result != None:
                    # logger.debug(f"{j=}, {result=}")
                    for class_id, bboxes in enumerate(result["track_bboxes"]):
                        model_class_id = class_id + (j * 10)
                        if len(bboxes) > 0:
                            best_id = -1
                            for _id, bbox in enumerate(bboxes):  
                                if bbox[0] == prev_track_ids[model_class_id]:
                                    best_id = _id
                                    break
                            # find the bbox with best confidence
                            if best_id < 0:
                                best_id = np.argmax(bboxes[:, -1])
                            # if j == 1:
                                # logger.debug(f"tracker[{j}], {result['tracks_boxes']=}")
                            #frame_tr.append(bboxes[best_id].tolist()[1:] + [object_class_id + (j * 10)])
                            frame_tr.append(bboxes[best_id].tolist()[1:] + [model_class_id])
                            prev_track_ids[model_class_id] = bboxes[best_id][0]
            tracking_results["tracks"][frame_id]=frame_tr

        #kick all nones
        tracking_results["tracks"]=[x for x in tracking_results["tracks"] if x is not None]

        json.dump(tracking_results, open(output_file_path, "w"))

        
def main_tracker_bytetrack_batch(
    trackers_config_and_checkpoints: list,
    # config_file,
    filename,
    # checkpoint,
    output_file_path: Path,
    device=None,
    class_names=[],
    additional_hash="",
):
    """Run tracking on a video.
    trackers: is list of tuples (config_file, checkpoint)
    additional_hash: is a string that will be added to the hash of the first frame
    """

    run_tracking, hash_hex = _should_do_tracking_based_on_hash(
        trackers_config_and_checkpoints, filename,
        output_file_path, additional_hash=additional_hash)

    models = []
    for tracker_config, tracker_checkpoint in trackers_config_and_checkpoints:
        # build the model from a config file and a checkpoint file
        model = init_model(tracker_config, str(tracker_checkpoint), device=device, verbose_init_params=False)
        models.append(model)



    imgs = mmcv.VideoReader(str(filename))
    frame_cnt = imgs.frame_cnt

    if run_tracking:
        progress = tools.ProgressPrinter(frame_cnt)
        tracking_results = {"tracks": [None]*int(frame_cnt), "hash": hash_hex, "class_names": class_names}
        # for frame_id, img in enumerate(imgs):
        N = len(imgs)
        batch_size = 3
        for i in range(0, N, batch_size):
            start_idx = i
            stop_idx = i + batch_size
            if stop_idx > N:
                stop_idx = N
            img = np.array(imgs[start_idx:stop_idx])
            frame_id = np.arange(start_idx,stop_idx,1)
            frame_tr = []
            # if not (frame_id % 50):
            #     logger.debug(
            #         f"Tracking on frame {frame_id}, {progress.get_progress_string(float(frame_id))}"
            #     )
            for j, tracker in enumerate(models):

                result = inference_mot_batch(
                    tracker, img, frame_id=frame_id
                )
 
                if result != None:
                    # logger.debug(f"{j=}, {result=}")
                    for object_class_id, bboxes in enumerate(result["track_bboxes"]):
                        if len(bboxes) > 0:
                            # find the bbox with best confidence
                            best_id = np.argmax(bboxes[:, -1])

                            # if j == 1:
                                # logger.debug(f"tracker[{j}], {result['tracks_boxes']=}")
                            frame_tr.append(bboxes[best_id].tolist()[1:] + [object_class_id + (j * 10)])
            tracking_results["tracks"][frame_id]=frame_tr

        #kick all nones
        tracking_results["tracks"]=[x for x in tracking_results["tracks"] if x is not None]

        json.dump(tracking_results, open(output_file_path, "w"))

        
def inference_mot_batch(model, img, frame_id):
    """Inference image(s) with the mot model.

    Args:
        model (nn.Module): The loaded mot model.
        img (str | ndarray): Either image name or loaded image.
        frame_id (int): frame id.

    Returns:
        dict[str : ndarray]: The tracking results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img, img_info=dict(frame_id=frame_id), img_prefix=None)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    else:
        # add information into dict
        data = dict(
            img_info=dict(filename=img, frame_id=frame_id), img_prefix=None)
    # build the data pipeline
    cfg.samples_per_gpu = 3
    
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result
