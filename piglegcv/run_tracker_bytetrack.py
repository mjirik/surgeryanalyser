# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser
import sys
import json
import shlex

import mmcv

from mmtrack.apis import inference_mot, init_model

def add_tracking_results(tracking_results, result):
    if result != None:
        frame_tr = []
        for i,tr in enumerate(result['track_bboxes']):
            if len(tr) > 0:
                frame_tr.append(tr[0].tolist()[1:] + [i])
        tracking_results['tracks'].append(frame_tr)

def main_tracker_bytetrack(commandline):

    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('filename', type=str, help='video file.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        "-out", "--output_dir", type=str, default="", help="Output directory."
    )

    # Parsing arguments
    #args = parser.parse_args()
    args = parser.parse_args(shlex.split(commandline))


    imgs = mmcv.VideoReader(args.filename)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # test and show/save the images
    tracking_results = {'tracks':[]}
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = osp.join(args.input, img)
        result = inference_mot(model, img, frame_id=i)

        add_tracking_results(tracking_results, result)
        
    json.dump(tracking_results, open(f'{args.output_dir}/tracks.json', 'w'))

