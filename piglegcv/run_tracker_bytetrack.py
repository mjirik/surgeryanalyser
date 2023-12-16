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

def add_tracking_results(tracking_results, result):
    if result != None:
        frame_tr = []
        for i, tr in enumerate(result['track_bboxes']):
            if len(tr) > 0:
                frame_tr.append(tr[0].tolist()[1:] + [i])
        tracking_results['tracks'].append(frame_tr)


def make_hash_from_model(model_file:Path):
    import hashlib
    hash_md5 = hashlib.md5()
    with open(model_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main_tracker_bytetrack(
        trackers_config_and_checkpoints: list,
        # config_file,
        filename,
        # checkpoint,
        output_file_path: Path,
        device=None,
        # score_thr=0.5,
        crop: list=[None, None, None, None],
        class_names=[],
):
    """Run tracking on a video.
    trackers: is list of tuples (config_file, checkpoint)
    """

    # make peceptual hash of first video frame
    imgs = mmcv.VideoReader(str(filename))
    first_frame = next(imgs)
    logger.debug(f"{first_frame.shape=}")
    phash = tools.phash_image(first_frame)
    del(imgs)

    hash_hex = phash
    models = []
    for tracker_config, tracker_checkpoint in trackers_config_and_checkpoints:
        # build the model from a config file and a checkpoint file
        model = init_model(tracker_config, str(tracker_checkpoint), device=device)
        models.append(model)
        hash_hex += make_hash_from_model(tracker_checkpoint)

    hash_length = len(hash_hex)
    run_tracking = True
    logger.debug(f"{hash_hex=}")
    if output_file_path.exists():
        try:
            data = json.load(open(output_file_path, 'r'))
            if ("hash" in data):
                distance = tools.phash_distance(hash_hex[:hash_length], data['hash'][:hash_length])

                if(data['hash'][hash_length:] == hash_hex[hash_length:]) and ():
                    run_tracking = False
                    logger.debug(f"Tracking results already exists ({distance=}). Skipping tracking.")
                else:
                    logger.debug(f"{data['hash']=}, ({distance=})")
        except Exception as e:
            logger.debug(f"Cannot read {Path(output_file_path).name}. Exception: {e}")
            
        # else:
        #     logger.debug(f"Hashes are different: {data['hash']} != {hash}")

    imgs = mmcv.VideoReader(str(filename))

    if run_tracking:
        tracking_results = {'tracks': [], "hash": hash_hex, "class_names": class_names}
        for i, img in enumerate(imgs):
            frame_tr = []
            if not (i % 50):
                logger.debug(f'Processing frame {i} by tracker')
            for j, tracker in enumerate(models):

                result = inference_mot(tracker, img[crop[0]:crop[1], crop[2]:crop[3], :], frame_id=i)

                if result != None:
                    # logger.debug(f"{j=}, {result=}")
                    for k, tr in enumerate(result['track_bboxes']):
                        if len(tr) > 0:
                            frame_tr.append(tr[0].tolist()[1:] + [k+(j*10)])
            # logger.debug(f"track_bboxes per frame {frame_tr}")
            tracking_results['tracks'].append(frame_tr)

            # add_tracking_results(tracking_results, result)

        json.dump(tracking_results, open(output_file_path, 'w'))

