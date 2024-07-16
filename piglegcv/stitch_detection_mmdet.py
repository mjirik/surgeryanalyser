import asyncio
from argparse import ArgumentParser
from mmdet.apis import (
    async_inference_detector,
    inference_detector,
    init_detector,
    show_result_pyplot,
)
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from pycocotools import mask as maskUtils

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy import stats
from mmcv import Config
from pathlib import Path
from skimage.transform import resize

from tools import load_json, save_json

import logging
import mmcv.utils

# logger = mmcv.utils.get_logger(name=__file__, log_level=logging.DEBUG)
from loguru import logger

# import mmdet
# mmdetection_path = Path(mmdet.__file__).parent.parent


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("img", help="Image file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=0.3, help="bbox score threshold"
    )

    args = parser.parse_args()
    return args



# def main(args):
def run_stitch_detection(img, json_file, confidence_treshold = 0.5, device="cuda"):

    cfg_path = Path("./stitch_detection_mmdet_config.py")
    checkpoint_path = (
        Path(__file__).parent / "resources/stitch_detection_models/model.pth"
    )
    logger.debug(f"{cfg_path=}, {cfg_path.exists()}")
    logger.debug(f"{checkpoint_path=}, {checkpoint_path.exists()}")

    cfg = Config.fromfile(str(cfg_path))

    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, str(checkpoint_path), device=device)
    # test a single image
    result = inference_detector(model, img)
    # show the results

    bbox_result = result
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)
    ]

    labels_best = []
    bboxes_best = []

    if len(bbox_result) > 0:
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)

        # Filter overlaping boxes
        OM = bbox_overlaps(bboxes, bboxes)
        # print(OM)
        stitches = []
        for i in range(OM.shape[0]):
            s = set()
            for j in range(OM.shape[1]):
                if OM[i, j] != 0.0:
                    s.add(i)
                    s.add(j)
            if len(s) > 0:
                if s not in stitches:
                    stitches.append(s)
                    # print(s)

        # print(stitches)

        bboxes_best_idx = set()
        for stitch in stitches:
            idx = np.array(list(stitch))
            idx_max = np.argmax(bboxes[idx, 4])
            bboxes_best_idx.add(idx[idx_max])

        if len(bboxes_best_idx) > 0:
            idx_best = np.array(list(bboxes_best_idx))
            bboxes_best = bboxes[idx_best].tolist()
            labels_best = labels[idx_best].tolist()
            
    # filter only confident bbox
    bboxes_filtered = []
    labels_filtered = []
    for box, label in zip(bboxes_best, labels_best):
        if box[4] > confidence_treshold:
            bboxes_filtered.append(box)
            labels_filtered.append(label)

    save_json({"stitch_labels": labels_filtered, "stitch_bboxes": bboxes_filtered}, json_file)

    logger.debug(f"number of filtered stitches = {len(bboxes_filtered)}")
    logger.debug(f"Stitch detection finished, boxes in: {json_file}")

    return bboxes_filtered, labels_filtered
    # return bboxes, labels


def run_stitch_analyser(
    img,
    bboxes,
    labels,
    expected_stitch_line,
    output_filename,
    basewidth=640,
    class_names=["<5", "5-10", "10-15", ">15"],
    bbox_color=[(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)],
):

    # uniform size ... basewidth
    # wpercent = 1.0
    wpercent = basewidth / float(img.shape[1])
    hsize = int((float(img.shape[0]) * float(wpercent)))

    img = resize(img, (hsize, basewidth), anti_aliasing=True)
    img = np.array(img * 255.0, dtype=np.uint8)

    bboxes = np.array(bboxes)
    labels = np.array(labels)

    if len(bboxes) > 0:
        bboxes[:, 0:4] *= wpercent

        img = imshow_det_bboxes(
            img[:, :, ::-1],
            bboxes,
            labels,
            class_names=class_names,
            bbox_color=bbox_color,
            text_color="white",
            mask_color=None,
            thickness=1,
            font_size=12,
            show=False,
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)

    # show_result_pyplot(model, args.img, result, score_thr=0.0)
    _, cols, _ = img.shape

    ########
    r_score = 0.0
    s_score = 0.0
    N = len(bboxes)
    p_score = 0.0
    if N > 0:
        p_score = 1. - (np.mean(labels) / 3.) #perpendicular score for 4 classes, 0 .. mean is 3, 1 .. mean is 0
        
    if N > 1:
        w = np.abs(bboxes[:, 0] - bboxes[:, 2])
        h = np.abs(bboxes[:, 1] - bboxes[:, 3])

        pad = 0.1 * np.max(np.array([w, h]), axis=0)

        x = bboxes[:, 0] + w / 2
        y1 = bboxes[:, 1] + pad
        y2 = bboxes[:, 3] - pad

        ax.plot(x, y1, "bo")
        ax.plot(x, y2, "bo")

        res1 = stats.linregress(x, y1)
        res2 = stats.linregress(x, y2)

        score1 = res1.rvalue**2
        score2 = res2.rvalue**2
        
        logger.debug(f"R-squared upper line: {score1:.3f}")
        logger.debug(f"R-squared lower line: {score2:.3f}")
        logger.debug(f"Slope upper line: {res1.slope:.3f}")
        logger.debug(f"Slope lower line: {res2.slope:.3f}")
        
        r_score = (score1 + score2) / 2.0
        s_score = 1.0 - (
            abs(np.arctan(res1.slope) - np.arctan(res2.slope)) / np.pi
        )  # np.arctan [-pi/2, pi/2]
                
        xx = np.array([0, cols])
        ax.plot(
            xx,
            res1.intercept + res1.slope * xx,
            "b:",
            label=f"r-score: {r_score:.2f}, s-score: {s_score:.3f}",
        )
        ax.plot(xx, res2.intercept + res2.slope * xx, "b:")

        # reference lines
        y0, y1, shift_px = expected_stitch_line
        ax.plot(xx, [wpercent * (y0 + shift_px), wpercent * (y1 + shift_px)], "k")
        ax.plot(xx, [wpercent * (y0 - shift_px), wpercent * (y1 - shift_px)], "k")

        plt.legend()
                
        
    # plt.show()
    plt.axis("off")
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)  # save image with result
    plt.close(fig)

    return {"r_score": r_score, "s_score": s_score, "N": N, "p_score" : p_score}


if __name__ == "__main__":

    print(Path(__file__))

    args = parse_args()

    image = np.array(Image.open(args.img))
    outputdir = "."
    i = 0

    bboxes_stitches, labels_stitches = run_stitch_detection(
        image, f"{outputdir}/stitch_detection_{i}.json"
    )
    run_stitch_analyser(
        image, bboxes_stitches, labels_stitches, f"{outputdir}/stitch_detection_{i}.jpg"
    )
