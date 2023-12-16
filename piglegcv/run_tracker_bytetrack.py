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
        class_names = None,
):
    """Run tracking on a video.
    trackers: is list of tuples (config_file, checkpoint)
    """

    hash = ""
    models = []
    for tracker_config, tracker_checkpoint in trackers_config_and_checkpoints:
        # build the model from a config file and a checkpoint file
        model = init_model(tracker_config, str(tracker_checkpoint), device=device)
        models.append(model)
        hash += make_hash_from_model(tracker_checkpoint)

    # nemělo by tady být spíš device="cuda" ? To ale nefunguje protože: RuntimeError: nms_impl: implementation for device cuda:0 not found.

    imgs = mmcv.VideoReader(str(filename))

    # build the model from a config file and a checkpoint file
    # model = init_model(config_file, str(checkpoint), device=device)
    # model = init_model(config_file, str(checkpoint), device=device)
    #print(model)

    # test and show/save the images
    run_tracking = True
    if output_file_path.exists():
        data = json.load(open(output_file_path, 'r'))
        if ("hash" in data) and (data['hash'] == hash):
            run_tracking = False
            logger.debug("Tracking results already exists. Skipping tracking.")



    tracking_results = {'tracks': [], "hash": hash, "class_names": class_names}
    for i, img in enumerate(imgs):
        frame_tr = []
        if not (i % 50):
            logger.debug(f'Processing frame {i} by tracker')
        for j, tracker in enumerate(trackers_config_and_checkpoints):

            result = inference_mot(tracker, img[crop[0]:crop[1], crop[2]:crop[3], :], frame_id=i)

            # if result != None:
            # logger.debug(f"{j=}, {result=}")
            for k, tr in enumerate(result['track_bboxes']):
                if len(tr) > 0:
                    frame_tr.append(tr[0].tolist()[1:] + [k+(j*10)])
        # logger.debug(f"track_bboxes per frame {frame_tr}")
        tracking_results['tracks'].append(frame_tr)

        # add_tracking_results(tracking_results, result)
        
    json.dump(tracking_results, open(output_file_path, 'w'))

