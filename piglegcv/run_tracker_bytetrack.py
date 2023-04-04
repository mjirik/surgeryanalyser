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

logger = logging.getLogger(__file__)

from mmtrack.apis import inference_mot, init_model

def add_tracking_results(tracking_results, result):
    if result != None:
        frame_tr = []
        for i,tr in enumerate(result['track_bboxes']):
            if len(tr) > 0:
                frame_tr.append(tr[0].tolist()[1:] + [i])
        tracking_results['tracks'].append(frame_tr)

def main_tracker_bytetrack(config_file, filename, checkpoint, output_dir="", device=None, score_thr=0.5, crop:list=[None, None, None, None]):
    
    # nemělo by tady být spíš device="cuda" ? To ale nefunguje protože: RuntimeError: nms_impl: implementation for device cuda:0 not found.

#     parser = ArgumentParser()
#     parser.add_argument('config', help='config file')
#     parser.add_argument('filename', type=str, help='video file.')
#     parser.add_argument(
#         '--score-thr',
#         type=float,
#         default=0.0,
#         help='The threshold of score to filter bboxes.')
# #     parser.add_argument(
# #         '--device', default=None, help='device used for inference')
#     parser.add_argument(
#         "-out", "--output_dir", type=str, default="", help="Output directory."
#     )

    # Parsing arguments
    #args = parser.parse_args()
#     args = parser.parse_args(shlex.split(commandline))
    print(f"device={device}")
    
    imgs = mmcv.VideoReader(str(filename))

    # build the model from a config file and a checkpoint file
    model = init_model(config_file, str(checkpoint), device=device)
    print(model)

    # test and show/save the images
    tracking_results = {'tracks':[]}
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = osp.join(args.input, img)
        result = inference_mot(model, img[crop[0]:crop[1], crop[2]:crop[3], :], frame_id=i)

        add_tracking_results(tracking_results, result)
        
    json.dump(tracking_results, open(f'{output_dir}/tracks.json', 'w'))

