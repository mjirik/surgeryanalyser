import os
import sys
import json
from argparse import ArgumentParser

import cv2

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from loguru import logger

def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 1 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]
    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results

def printKeypoints(pose_results, _file):
    if len(pose_results) > 0:
        all_points = []
        for keyp in pose_results[0]['keypoints']:
            points = ' '.join(['{}'.format(x) for x in keyp])
            all_points.append(points)
        _file.write('{}\n'.format(','.join(all_points)))
    else:
        _file.write('None\n')

def save_json(data: dict, output_json: str):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as output_file:
        json.dump(data, output_file)




###############################
def main_mmpose(filename, outputdir):
    
    det_config = 'cascade_rcnn_x101_64x4d_fpn_1class.py'
    det_checkpoint = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'
    device = 'cuda:0'
    det_model = init_detector(
        det_config, det_checkpoint, device=device)
    # build the pose model from a config file and a checkpoint file
    pose_config = 'res50_onehand10k_256x256.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device)

    dataset = pose_model.cfg.data['test']['type']

    cap = cv2.VideoCapture(filename)
    #assert cap.isOpened(), f'Faild to load video file {args.video_path}'


    save_out_video = False
    video_name = '{}/hand_poses.mp4'.format(outputdir)
    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(video_name, fourcc, fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    det_cat_id = 1
    bbox_thr = 0.3
    return_heatmap = False
    output_layer_names = None
    kpt_thr = 0.5
    radius = 5
    thickness = 3
    cv2.setNumThreads(2)
    #out_file = open('{}/hand_poses.txt'.format(outputdir), 'w')
    hand_poses = []
    
    frame_id = -1
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        frame_id += 1
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

        vis_img = img
        pose_data = []
        pose_results = None

        if (len(person_results) > 0) and (person_results[0]['bbox'][4] > 0.5):
            #print(person_results)
            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                bbox_thr=bbox_thr,
                format='xyxy',
                dataset=dataset,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
            
            #printKeypoints(pose_results, out_file)
            if len(pose_results) > 1:
                pose_data = [pose_results[0]['keypoints'].tolist(), pose_results[1]['keypoints'].tolist()]
        hand_poses.append(pose_data)
        
        if not(frame_id % 10):
            logger.debug(f'Frame {frame_id} processed!')

        if save_out_video and pose_results:
            vis_img = vis_pose_result(
                pose_model,
                img,
                pose_results,
                dataset=dataset,
                kpt_score_thr=kpt_thr,
                radius=radius,
                thickness=thickness,
                show=False)

            videoWriter.write(vis_img)

    cap.release()
    if save_out_video:
        videoWriter.release()

    save_json({"hand_poses": hand_poses}, os.path.join(outputdir, "hand_poses.json"))
    
if __name__ == '__main__':
    main()
