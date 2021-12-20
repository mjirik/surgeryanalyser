import os
import cv2
import json
import torch
import argparse
import numpy as np
import albumentations as A

import shlex
from pyzbar.pyzbar import decode

from detectron2.data import transforms as T

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from extern.sort import sort

CFG = {
    "output_dir": "./__OUTPUT__/",
    "SORT": {"max_age": 4, "min_hits": 6, "iou_threshold": 0.0},  # int  # int  # float
    "TRACKING": {
        "THRESHOLD": {
            "thr_of_score_1": 0.95,  # upper threshold defined as "a minimal score of detection to be used, when the
                                     # maximal score is equal to 1"
            "the_of_score_x": 0.20,  # lower threshold defined as "a minimal score of detection to be used, when the
                                     # maximal score is equal to x" -> all detections with score lower than this value
                                     # are newer used
            "x": 0.50   # when maximal score of detection is equal to this value (in current frame), "t_of_score_x"
                        # defines its minimal score for all others detections that are required to be used for tracking
        }
    }
}
FLOAT_EPSILON = 1e-5


def custom_img_preprocessing_test(image):
    # init the local variables
    domain_adapt = CFG['R-CNN']['INPUT']['AUGMENTATIONS']['DOMAIN_ADAPT']
    augm_type = str(domain_adapt['type']).upper()
    transforms_alb = list()

    # domain augmentations
    if (augm_type not in ["NONE", ""]) and (domain_adapt[augm_type]['prob'] >= 1.0 - FLOAT_EPSILON):
        if domain_adapt['type'] == "FDA":
            transforms_alb.append(
                A.FDA(
                    domain_adapt['ref_img'],
                    beta_limit=domain_adapt['FDA']['beta_limit'],
                    p=1.0
                )
            )
        elif domain_adapt['type'] == "histogram_matching":
            blend_ratio = sum(domain_adapt['HISTOGRAM_MATCHING']['blend_ratio']) / 2

            transforms_alb.append(
                A.HistogramMatching(
                    domain_adapt['ref_img'],
                    blend_ratio=(blend_ratio, blend_ratio),
                    p=1.0
                )
            )
        elif domain_adapt['type'] == "pixel_distribution_adapt":
            blend_ratio = sum(domain_adapt['PIXEL_DISTRIBUTION_ADAPT']['blend_ratio']) / 2

            transforms_alb.append(
                A.PixelDistributionAdaptation(
                    domain_adapt['ref_img'],
                    blend_ratio=(blend_ratio, blend_ratio),
                    transform_type=domain_adapt['PIXEL_DISTRIBUTION_ADAPT']['transform_type'],
                    p=1.0
                )
            )

    # resize an image and apply the transforms
    new_width, new_height = 0, 0
    if CFG['R-CNN']['INPUT']['RESIZE']['type'].lower() == "relative":
        new_height = int(image.shape[0] * CFG['R-CNN']['INPUT']['RESIZE']['RELATIVE']['ratio'])
        new_width = int(image.shape[1] * CFG['R-CNN']['INPUT']['RESIZE']['RELATIVE']['ratio'])
    elif CFG['R-CNN']['INPUT']['RESIZE']['type'].lower() == "img_edge":
        resize_base = CFG['R-CNN']['INPUT']['RESIZE']['IMG_EDGE']

        # check shortest edge of the input image and resize it if it is higher than maximum of sizes, or lower than
        # minimum of sizes
        if resize_base['type'] == "shortest":
            if image.shape[0] < image.shape[1]:
                if image.shape[0] > max(resize_base['sizes']):
                    new_height = max(resize_base['sizes'])
                elif image.shape[0] < min(resize_base['sizes']):
                    new_height = min(resize_base['sizes'])
                else:
                    new_height = 0
                new_width = new_height * image.shape[1] / image.shape[0]
            else:
                if image.shape[1] > max(resize_base['sizes']):
                    new_width = max(resize_base['sizes'])
                elif image.shape[1] < min(resize_base['sizes']):
                    new_width = min(resize_base['sizes'])
                else:
                    new_width = 0
                new_height = new_width * image.shape[0] / image.shape[1]
        elif resize_base['type'] == "height":
            if image.shape[0] > max(resize_base['sizes']):
                new_height = max(resize_base['sizes'])
            elif image.shape[0] < min(resize_base['sizes']):
                new_height = min(resize_base['sizes'])
            else:
                new_height = 0
            new_width = new_height * image.shape[1] / image.shape[0]
        elif resize_base['type'] == "width":
            if image.shape[1] > max(resize_base['sizes']):
                new_width = max(resize_base['sizes'])
            elif image.shape[1] < min(resize_base['sizes']):
                new_width = min(resize_base['sizes'])
            else:
                new_width = 0
            new_height = new_width * image.shape[0] / image.shape[1]
        else:
            raise Exception(f"Wrong type of resizing image for IMG_EDGE. You have {resize_base['type']}")
    else:
        raise Exception(f"Wrong type of resizing image. You have {CFG['R-CNN']['INPUT']['RESIZE']['type']}")

    # resize an image
    transforms = []
    if int(new_height) != 0:
        # the image needs to be resized
        image, transforms = T.apply_transform_gens([
            T.Resize((
                int(new_height),
                int(new_width)
            ))
        ], image)

    if len(transforms_alb) != 0:
        # albumentation transforms
        image = A.Compose(transforms_alb)(image=image)["image"]

    return image, transforms


class CustomPredictor(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR", "L"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]  # needed for recomputing the boxes positions

            # preprocess an image
            image, _ = custom_img_preprocessing_test(original_image)

            if self.input_format == "L":
                image = torch.as_tensor(np.ascontiguousarray(image))
            else:
                image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            return predictions


def merge_config():
    if args.config is not None:
        with open(args.config, "r") as json_file:
            CFG.update(json.load(json_file))


def get_detectron_cfg() -> CfgNode:
    """
    Merge the local config file with the detectron2 one.

    :return: detectron2 config node
    """

    # initialize the configuration
    cfg = get_cfg()

    # model specification
    cfg.merge_from_file(
        model_zoo.get_config_file(f"COCO-Detection/{CFG['R-CNN']['model']}.yaml")
    )
    cfg.MODEL.WEIGHTS = CFG["R-CNN"]["weights"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = CFG["DATA"]["num_classes"]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.25

    # proposals of bounding boxes
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = CFG["R-CNN"]["ANCHOR"]["aspect_ratios"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = CFG["R-CNN"]["ANCHOR"]["sizes"]

    # input image
    cfg.INPUT.FORMAT = CFG["R-CNN"]["INPUT"]["format"]

    # which dataset shall be used for training and validation
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()

    # evaluation
    cfg.TEST.DETECTIONS_PER_IMAGE = CFG["DATA"]["max_dets"]

    # dataloader
    cfg.DATALOADER.NUM_WORKERS = CFG["DATA"]["num_workers"]

    # batch size
    cfg.SOLVER.IMS_PER_BATCH = CFG["SOLVER"]["batch_size"]

    ## output path
    cfg.OUTPUT_DIR = CFG["output_dir"]
    #prefix = "" if (CFG["OUTPUT"]["prefix"] == "") else f"{CFG['OUTPUT']['prefix']}_"
    #suffix = "" if (CFG["OUTPUT"]["suffix"] == "") else f"_{CFG['OUTPUT']['suffix']}"
    #cfg.OUTPUT_DIR = os.path.join(
        #CFG["output_dir"], "tracker", f"{prefix}{CFG['R-CNN']['model']}{suffix}"
    #)

    ## check if output path exists
    #output_orig = cfg.OUTPUT_DIR
    #idx = 1
    #while os.path.exists(cfg.OUTPUT_DIR):
        #cfg.OUTPUT_DIR = f"{output_orig}__{idx}"
        #idx += 1

    return cfg


def read_img(img_name: str):
    if CFG["R-CNN"]["INPUT"]["format"] == "L":  # greyscale
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_name)

    # preprocess an image - �t is done by predictor
    # img, _ = custom_img_preprocessing_test(img)

    return img


def save_json(data: dict, output_json: str):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as output_file:
        json.dump(data, output_file)




def tracking_sort(
    predictor: CustomPredictor, tracker: [sort.Sort], filename: str, output_dir: str
):
    track_id_last = [1 for _ in range(CFG["DATA"]["num_classes"])]
    final_tracks = list()

    thr_a = (CFG["TRACKING"]["THRESHOLD"]["thr_of_score_1"] - CFG["TRACKING"]["THRESHOLD"]["the_of_score_x"]) / \
            (1 - CFG["TRACKING"]["THRESHOLD"]["x"])
    thr_b = CFG["TRACKING"]["THRESHOLD"]["thr_of_score_1"] - thr_a

    cap = cv2.VideoCapture(str(filename))
    frame_id = -1
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        frame_id += 1

        #predict
        outputs = predictor(img)
        outputs = outputs["instances"].to("cpu")

        # filter unwanted detections (only when any detection exists)
        dets = outputs.pred_boxes.tensor.numpy()
        scores = outputs.scores.numpy()
        classes = outputs.pred_classes.numpy()

        # divide outputs by its predicted class
        split_dets = [list() for _ in range(CFG["DATA"]["num_classes"])]     # [[]] * CFG["DATA"]["num_classes"] is not
        split_scores = [list() for _ in range(CFG["DATA"]["num_classes"])]   # working...
        for i, cat in enumerate(classes):
            split_dets[cat].append(dets[i])
            split_scores[cat].append(scores[i])

        # track each category
        filtered_tracks = [list() for _ in range(CFG["DATA"]["num_classes"])]
        for cat in range(CFG["DATA"]["num_classes"]):
            # check if prediction exists for this category
            if len(split_scores[cat]) == 0:
                continue

            # filter predictions
            max_score = max(split_scores[cat])
            if len(split_dets[cat]) > 0:
                det = []
                for s, d in zip(split_scores[cat], split_dets[cat]):
                    if s >= round(thr_a * max_score + thr_b, 3) and s >= CFG["TRACKING"]["THRESHOLD"]["the_of_score_x"]:
                        det.append(d)

                dets = np.array(det)

            # update SORT
            tracks = tracker[cat].update(dets if len(dets) > 0 else np.empty((0, 5)))

            # check if the last track id is in the list of all tracks for current image
            skip_wrong_tracks = True if track_id_last[cat] in [t[4] for t in tracks] else False

            # actualize the track id when it was lost with Kalman Filter
            if (len(tracks) == 1) and skip_wrong_tracks is False:
                track_id_last[cat] = tracks[0][4]

            # tracks are in the format: [x_tl, y_tl, x_br, y_br, track_id], where tl is top-left and br is bottom-right
            for t, track in enumerate(tracks):
                if (skip_wrong_tracks is True) and (track[4] != track_id_last[cat]):
                    continue

                # save the tracks after filtering
                filtered_tracks[cat].append(track.tolist() + [cat])

        # store the final tracks to the list
        all_tracks = list()
        for cat in range(len(filtered_tracks)):
            all_tracks += filtered_tracks[cat]
        final_tracks.append(all_tracks)
        
        if not(frame_id % 10):
            logger.debug(f'Frame {frame_id} processed!')

    # save the final tracks to the json file
    save_json({"tracks": final_tracks}, os.path.join(output_dir, "tracks.json"))



#if __name__ == "__main__":
def main_tracker(commandline):
    print("main_tracker: initiated")
    # Parse commandline
    parser = argparse.ArgumentParser(
        description="Tracking the objects using MOT methods."
    )

    # Optional arguments
    parser.add_argument(
        "-pre", "--prefix", type=str, default="", help="Prefix for the output."
    )
    parser.add_argument(
        "-suf", "--suffix", type=str, default="", help="Suffix for the output."
    )
    parser.add_argument(
        "-out", "--output_dir", type=str, default="", help="Output directory."
    )

    # Positional arguments
    parser.add_argument(
        "model_dir", type=str, help="Path to the directory with a model."
    )
    parser.add_argument("filename", type=str, help="video file.")

    # Parsing arguments
    args = parser.parse_args(shlex.split(commandline))

    # ==================================================================================================================

    # update a config
    with open(os.path.join(args.model_dir, "config.json"), "r") as json_file:
        CFG.update(json.load(json_file))

    if os.path.exists(os.path.join(args.model_dir, "last_checkpoint")):
        with open(
            os.path.join(args.model_dir, "last_checkpoint"), "r"
        ) as checkpoint_file:
            CFG["R-CNN"]["weights"] = os.path.join(
                args.model_dir, checkpoint_file.readline()
            )

    CFG['R-CNN']['INPUT']['AUGMENTATIONS']['DOMAIN_ADAPT']['ref_img'] = \
        [os.path.join(args.model_dir, i) for i in CFG['R-CNN']['INPUT']['AUGMENTATIONS']['DOMAIN_ADAPT']['ref_img']]

    # update the optional arguments
    if args.prefix is not "":
        CFG["OUTPUT"]["prefix"] = args.prefix

    if args.suffix is not "":
        CFG["OUTPUT"]["suffix"] = args.suffix

    if args.output_dir is not "":
        CFG["output_dir"] = args.output_dir

    # get the detectron2 configuration and create an output directory
    cfg = get_detectron_cfg()


    #os.makedirs(cfg.OUTPUT_DIR)

    # save the used configuration (for training, testing...)
    save_json(CFG, os.path.join(cfg.OUTPUT_DIR, "config.json"))

    # initialize the predictor
    predictor = CustomPredictor(cfg)

    # use the SORT method for tracking the objects
    mot_tracker = [sort.Sort(
        max_age=CFG["SORT"]["max_age"],
        min_hits=CFG["SORT"]["min_hits"],
        iou_threshold=CFG["SORT"]["iou_threshold"],
    ) for _ in range(CFG["DATA"]["num_classes"])]

    # get tracks
    tracking_sort(predictor, mot_tracker, args.filename, cfg.OUTPUT_DIR)
