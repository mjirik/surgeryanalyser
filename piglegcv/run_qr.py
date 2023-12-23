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
import tools
from typing import Optional


def get_bboxes(img, device="cpu", image_file:Optional[Path]=None,
               calibration_micro_thr:float=0.5
              ):
    single_model_path = (
        Path(__file__).parent / "resources/single_image_detector/mdl_sid_2.pth"
        # Path(__file__).parent / "resources/single_image_detector/mdl.pth"
    )
    _model = torch.load(single_model_path, map_location=torch.device(device))
    single_image_model = _model["model"]
    single_image_model.cfg = _model["my_params"]

    bboxes, masks = inference_detector(single_image_model, img)
    if image_file is not None:
        logger.debug("saving single_image_detector result")
        single_image_model.show_result(
            img,
            bboxes,
            # masks,
            # score_thr=0.3,
            show=False,
            out_file=str(image_file)
        )
    # -1: incision area
    # 0: scene area
    # 1: ?
    # 2: QR code
    # 3: ?
    # 4: ?
    # 5: micro qr code
    # for cls_id, bboxes_class in enumerate(bboxes):
    #     logger.debug(f"{cls_id=}, {bboxes_class=}")

    bboxes_incision_area = bboxes[0]
    bboxes_incision_area = tools.sort_bboxes(bboxes_incision_area)
    
    ia_threshold = 0.8
    ia_filter = bboxes_incision_area[:, -1] > ia_threshold
    bboxes_incision_area = bboxes_incision_area[ia_filter]
    

    scene_area_threshold = 0.35
    scene_area_bboxes = bboxes[1]
    scene_area_bboxes = tools.sort_bboxes(scene_area_bboxes)
    if scene_area_bboxes.shape[0] > 0:
        bbox_scene_area = scene_area_bboxes[0]
        if bbox_scene_area[-1] < scene_area_threshold:
            bbox_scene_area = None

    else:
        bbox_scene_area = None

    qr_threshold = 0.85
    if bboxes[3].shape[0] > 0:
        logger.debug(bboxes[3])
        bboxes_qr = bboxes[3][:2]
        qr_filter = bboxes_qr[:, -1] > qr_threshold
        bboxes_qr = bboxes_qr[qr_filter]

        qr_mask = masks[3][0]
        side_length = math.sqrt(np.count_nonzero(qr_mask == True))
    else:
        bboxes_qr = []
        side_length = None
    if bboxes[5].shape[0] > 0:
        bboxes_calibration_micro, masks_calibration_micro = tools.sort_bboxes_and_masks_by_confidence(bboxes[5], masks[5])
        if bboxes_calibration_micro[0][-1] > calibration_micro_thr:
            logger.debug(f"micro calibration detected")
            micro_side_length = 2.0 * math.sqrt(np.count_nonzero(masks_calibration_micro[0] == True) / np.pi)
            logger.debug(f"{bboxes_calibration_micro=}, {micro_side_length=}")
        else:
            bboxes_calibration_micro = []
            micro_side_length = None
            
    else:
        bboxes_calibration_micro = []
        micro_side_length = None
    #     bboxes_qr = bboxes[3][:2] if bboxes[3].shape[0] > 0 else None



    return bboxes_incision_area, bbox_scene_area, bboxes_qr, side_length, bboxes_calibration_micro, micro_side_length


def bbox_info_extraction_from_frame(img, qreader=None, device="cpu", image_file:Optional[Path]=None):
    img = np.asarray(img)
    width = img.shape[1]
    # Todo Viktora
    bboxes_incision_area, bbox_scene_area, bboxes_qr, qr_side_length, bboxes_calibration_micro, micro_side_length = get_bboxes(
        img, device=device, image_file=image_file
    )

    if qreader is None:

        qreader = QReader()

    detected_qr_codes = qreader.detect_and_decode(image=img, return_bboxes=True)
    pix_size_best = None
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
                qr_scissors_frame_detected = True
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
                qr_bbox = [
                    [
                        int(point.x / resize_factor + bbox[0]),
                        int(point.y / resize_factor + bbox[1]),
                    ]
                    for point in oneqr.polygon
                ]

                # debug only
                # cv2.drawContours(img, [np.asarray(box)], 0, (0, 255, 0), 2)
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # cv2.imshow("image", img)
                # cv2.waitKey(0)
                # cv2.imwrite(filename='_image.jpeg', img=img)
                # debug only

                a = np.array(qr_bbox[0])
                b = np.array(qr_bbox[1])
                pix_size_best = qr_size / np.linalg.norm(a - b)
        output = {}
    # todo use the pigleg holder detection based estimator
    pix_size_single_frame_detector_m = (
        qr_size / qr_side_length if qr_side_length else None
    )

    qr_data = {
        "is_microsurgery": False
    }

    pix_size_method = "video size estimation"
    if is_detected:
        pix_size_method = "QR"
    elif len(bboxes_calibration_micro) > 0:
        pix_size_method = "micro calibration"
        pix_size_best = 0.006 / micro_side_length
        qr_data["is_microsurgery"] = True
        is_detected = True
    elif len(bboxes_qr) > 0:
        pix_size_method = "pix_size_single_frame_detector_m"
        pix_size_best = pix_size_single_frame_detector_m

    qr_data["pix_size_method"] = pix_size_method
    if True:
        # pigleg_holder_width [m] - usually it takes around half of the image width
        scene_size = 0.300  # [m]
        size_by_scene = scene_size / width

    qr_data["is_detected"] = is_detected
    qr_data["box"] = qr_bbox
    qr_data["pix_size"] = pix_size_best
    qr_data["incision_bboxes"] = np.asarray(bboxes_incision_area).tolist()
    qr_data["qr_size"] = qr_size
    qr_data["size_by_scene"] = size_by_scene
    qr_data["text"] = qr_text
    qr_data["pix_size_single_frame_detector_m"] = pix_size_single_frame_detector_m
    qr_data["bbox_scene_area"] = (
        np.asarray(bbox_scene_area).tolist() if bbox_scene_area is not None else None
    )
    qr_data["bbox_micro_calibration"] = np.asarray(bboxes_calibration_micro).tolist()
    qr_data["qr_scissors_frame_detected"] = qr_scissors_frame_detected
    qr_data["qr_bboxes_SID"] = (
        np.asarray(bboxes_qr).tolist() if bboxes_qr is not None else None
    )
    qr_data["scene_width_m"] = None if pix_size_best is None else width * pix_size_best

    logger.debug(qr_data)

    return qr_data
    # pix_size_best, qr_size, is_detected, qr_bbox, qr_text, qr_scissors_frame_detected #, bbox_scene_area, bboxes_incision_area


# TODO add device
def main_qr(filename, output_dir, device):
    """
    Detect QR cod in video and detect frames with scissors of the QR code.
    :param filename:
    :param output_dir:
    :return:
    """
    logger.debug("looking for qr code...")
    qreader = QReader()

    cap = cv2.VideoCapture(str(filename))
    pix_size = 1.0
    qr_size = 0.027
    is_detected = False
    box = []
    qr_text = None
    qr_scissors_frames = []
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

            # try read QR code
            logger.debug(f"frame={i}")

            qr_data = bbox_info_extraction_from_frame(
                img, device=device, qreader=qreader
            )
            qr_scissors_frame_detected = qr_data["qr_scissors_frame_detected"]
            if qr_scissors_frame_detected:
                qr_scissors_frames.append(i)

    qr_data = bbox_info_extraction_from_frame(img, qreader, device=device)

    qr_data["qr_scissors_frames"] = qr_scissors_frames

    # save QR to the json file
    json_file = Path(output_dir) / "meta.json"
    print(f"prepared to save to file {str(json_file)}")
    logger.debug(f"prepared to save to file {str(json_file)}")
    save_json({"qr_data": qr_data}, json_file)
    return qr_data


if __name__ == "__main__":

    main_qr(sys.argv[1], sys.argv[2], device="cpu")
