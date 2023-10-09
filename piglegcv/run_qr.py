import sys
import os
import cv2
import json
import numpy as np
from loguru import logger
from tools import save_json, load_json
from pathlib import Path
from qreader import QReader
import math
from mmdet.apis import inference_detector
import torch
import pprint


def get_bboxes(img):
    single_model_path = Path(__file__).parent / "resources/single_image_detector/mdl.pth"
    single_image_model = torch.load(single_model_path)["model"]
    single_image_model_cfg = torch.load(single_model_path)["my_params"]
    single_image_model.cfg = single_image_model_cfg

    bboxes, masks = inference_detector(single_image_model, img)

    bboxes_inicision_area = bboxes[0]
    
    scene_area_threshold = 0.35
    if bboxes[1].shape[0] > 0:
        bbox_scene_area = bboxes[1][0]
        if bbox_scene_area[-1] < scene_area_threshold:
            bbox_scene_area = None
        
    else:
        bbox_scene_area = None
        
    qr_threshold = 0.9
    if bboxes[3].shape[0] > 0:
        logger.debug(bboxes[3])
        bboxes_qr = bboxes[3][:2]
        qr_filter = bboxes_qr[:, -1] > qr_threshold
        bboxes_qr = bboxes_qr[qr_filter]
        
        qr_mask = masks[3][0]
        side_length = math.sqrt(np.count_nonzero(qr_mask == True))
    else:
        bboxes_qr = None
        side_length = None
#     bboxes_qr = bboxes[3][:2] if bboxes[3].shape[0] > 0 else None

    ia_threshold = 0.8
    ia_filter = bboxes_inicision_area[:, -1] > ia_threshold
    bboxes_inicision_area = bboxes_inicision_area[ia_filter]

 


    return bboxes_inicision_area, bbox_scene_area, bboxes_qr, side_length

def bbox_info_extraction_from_frame(img, qreader=None):
    img = np.asarray(img)
    width = img.shape[1]
    # Todo Viktora
    bboxes_incision_area, bbox_scene_area, bboxes_qr, qr_side_length = get_bboxes(img)

    if qreader is None:
        
        qreader = QReader()
    
    detected_qr_codes = qreader.detect_and_decode(image=img, return_bboxes=True)
    pix_size_best = 1.0
    qr_size = 0.027
    is_detected = False
    qr_bbox = []
    qr_text = None
    qr_scissors_frame_detected = False

    for bbox, _oneqr in detected_qr_codes:
        if _oneqr != None:
            oneqr, resize_factor = _oneqr
            txt = oneqr.data.decode("utf8")
            logger.debug(f"qr code text = '{txt}'")
            if txt == "Resolution 30 mm":
                qr_size = 0.030
                qr_text = txt
            elif txt == "QR scale pigleg":
                qr_size = 0.027
                qr_text = txt
            elif txt == "Scissors 30 mm":
                qr_scissors_frame_detected=True
                if qr_text is None:
                    # Use only if no Scale QR code was detected
                    qr_size = 0.030
                    qr_text = txt
            else:
                logger.debug(f"Unknown QR code with text='{txt}'")
                continue
            if not is_detected:
                is_detected = True

                # qreader returns detection bbox to input image, pyzbar reactangle and polygon to the bbox crop resized by resize_factor -> 
                # qr code polygon in input image = polygon / resize_facor + bbox 
                qr_bbox = [[int(point.x / resize_factor + bbox[0]), int(point.y / resize_factor + bbox[1])] for point in oneqr.polygon]

                # debug only
                #cv2.drawContours(img, [np.asarray(box)], 0, (0, 255, 0), 2)
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #cv2.imshow("image", img)
                #cv2.waitKey(0)
                #cv2.imwrite(filename='_image.jpeg', img=img)
                # debug only

                a = np.array(qr_bbox[0])
                b = np.array(qr_bbox[1])
                pix_size_best = qr_size / np.linalg.norm(a-b)
        output = {}
    # todo use the pigleg holder detection based estimator
    qr_data = {}
    qr_data["pix_size_method"] = "QR" if is_detected else "video size estimation"
    if True:
        # pigleg_holder_width [m] - usually it takes around half of the image width
        scene_size = 0.300 # [m]
        size_by_scene = scene_size / width


    qr_data['is_detected'] = is_detected
    qr_data['box'] = qr_bbox
    qr_data['pix_size'] = pix_size_best
    qr_data['qr_size'] = qr_size
    qr_data['size_by_scene'] = size_by_scene
    qr_data['text'] = qr_text
    qr_data["pix_size_single_frame_detector_m"] = qr_size / qr_side_length if qr_side_length else None
    qr_data["bbox_scene_area"] = np.asarray(bbox_scene_area).tolist() if bbox_scene_area is not None else None
    qr_data["qr_scissors_frame_detected"] = qr_scissors_frame_detected

    logger.debug(pprint.pformat(qr_data))

    return qr_data
    #pix_size_best, qr_size, is_detected, qr_bbox, qr_text, qr_scissors_frame_detected #, bbox_scene_area, bboxes_incision_area


def main_qr(filename, output_dir):
    logger.debug("looking for qr code...")
    qreader = QReader()
    
    cap = cv2.VideoCapture(str(filename))
    pix_size = 1.0
    qr_size = 0.027
    is_detected = False
    box = []
    qr_text = None
    qr_scissors_frames= []
    i = -1
    first_img = False
    image_processing_step = 10

    # TODO find another way how to detect scissors
    while cap.isOpened():
        i += 1
        ret = cap.grab()
        
        if not ret:
            break
        
        if not i % image_processing_step:
            _, img = cap.retrieve()
        
            # if not first_img:
            #     first_img = True
            #     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
            #     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

            #try read QR code
            logger.debug(f"frame={i}")

            qr_data = bbox_info_extraction_from_frame(img, qreader)
            qr_scissors_frame_detected = qr_data["qr_scissors_frame_detected"]
            if qr_scissors_frame_detected:
                qr_scissors_frames.append(i)

    qr_data = bbox_info_extraction_from_frame(img, qreader)

    qr_data['qr_scissors_frames'] = qr_scissors_frames

    # save QR to the json file
    json_file = Path(output_dir) / "meta.json"
    print(f"prepared to save to file {str(json_file)}")
    logger.debug(f"prepared to save to file {str(json_file)}")
    save_json({"qr_data": qr_data}, json_file)
    return qr_data
   
if __name__ == '__main__':

    main_qr(sys.argv[1], sys.argv[2])
