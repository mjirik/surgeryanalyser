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




def main_tracker_bytetrack(
        trackers: list,
        # config_file,
        filename,
        # checkpoint,
        output_file_path: Path,
        # device=None,
        # score_thr=0.5,
        crop: list=[None, None, None, None]):
    
    # nemělo by tady být spíš device="cuda" ? To ale nefunguje protože: RuntimeError: nms_impl: implementation for device cuda:0 not found.

    imgs = mmcv.VideoReader(str(filename))

    # build the model from a config file and a checkpoint file
    # model = init_model(config_file, str(checkpoint), device=device)
    # model = init_model(config_file, str(checkpoint), device=device)
    #print(model)

    # test and show/save the images
    tracking_results = {'tracks': []}
    for i, img in enumerate(imgs):
        frame_tr = []
        if not (i % 50):
            logger.debug(f'Processing frame {i} by tracker')
        for j, tracker in enumerate(trackers):

            result = inference_mot(tracker, img[crop[0]:crop[1], crop[2]:crop[3], :], frame_id=i)

            # if result != None:
            for k, tr in enumerate(result['track_bboxes']):
                if len(tr) > 0:
                    frame_tr.append(tr[0].tolist()[1:] + [i+j*10])
        tracking_results['tracks'].append(frame_tr)

        # add_tracking_results(tracking_results, result)
        
    json.dump(tracking_results, open(output_file_path, 'w'))

