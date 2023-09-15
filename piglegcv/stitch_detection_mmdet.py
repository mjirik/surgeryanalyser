import asyncio
from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
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
logger = mmcv.utils.get_logger(name=__file__, log_level=logging.DEBUG)

#import mmdet
#mmdetection_path = Path(mmdet.__file__).parent.parent

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')

    args = parser.parse_args()
    return args


#def main(args):
def run_stitch_detection(img, json_file, device="cuda"):


    cfg = Config.fromfile('./stitch_detection_mmdet_config.py')

    checkpoint_path = Path(__file__).parent / "resources/stitch_detection_models/model.pth"

    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, str(checkpoint_path), device=device)
    # test a single image
    result = inference_detector(model, img)
    # show the results

    bbox_result = result
    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]


    labels_best = []
    bboxes_best = []

    if len(bbox_result) > 0:
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)

        #Filter overlaping boxes
        OM = bbox_overlaps(bboxes, bboxes)
        #print(OM)
        stitches = []
        for i in range(OM.shape[0]):
            s = set()
            for j in range(OM.shape[1]):
                if OM[i,j] != 0.0:
                    s.add(i)
                    s.add(j)
            if len(s) > 0:
                if s not in stitches:
                    stitches.append(s)
                    #print(s)

        #print(stitches)

        bboxes_best_idx = set()
        for stitch in stitches:
            idx = np.array(list(stitch))
            idx_max = np.argmax(bboxes[idx,4])
            bboxes_best_idx.add(idx[idx_max])

        if len(bboxes_best_idx) > 0:
            idx_best = np.array(list(bboxes_best_idx))
            bboxes_best = bboxes[idx_best].tolist()
            labels_best = labels[idx_best].tolist()

    save_json({"stitch_labels": labels_best,
               "stitch_bboxes": bboxes_best
              }, json_file)

    logger.debug("stitch detection finished "+json_file)

    return bboxes_best, labels_best
    #return bboxes, labels




def run_stitch_analyser(img, bboxes, labels, output_filename, basewidth=640, class_names=['<5', '5-10', '10-15','>15'], bbox_color=[(0,255,0),(0,255,255), (0,165,255), (0,0,255)]):


    #uniform size ... basewidth
    #wpercent = 1.0
    wpercent = (basewidth/float(img.shape[1]))
    hsize = int((float(img.shape[0])*float(wpercent)))

    img = resize(img, (hsize, basewidth),anti_aliasing=True)
    img = np.array(img * 255.0, dtype=np.uint8)

    bboxes = np.array(bboxes)
    labels = np.array(labels)

    if len(bboxes)>0:
        bboxes[:,0:4] *= wpercent

        img = imshow_det_bboxes(img, bboxes, labels, class_names=class_names,
                        bbox_color=bbox_color,
                        text_color='white',
                        mask_color=None,
                        thickness=1,
                        font_size=12, show=False)


    plt.imshow(img)

    #show_result_pyplot(model, args.img, result, score_thr=0.0)

    ########
    if len(bboxes)>1:
        x1 = np.concatenate([bboxes[:,0],bboxes[:,2]])
        y1 = np.concatenate([bboxes[:,1],bboxes[:,1]])
        res1 = stats.linregress(x1, y1)
        x2 = np.concatenate([bboxes[:,0],bboxes[:,2]])
        y2 = np.concatenate([bboxes[:,3],bboxes[:,3]])
        res2 = stats.linregress(x2, y2)

        score1 = res1.rvalue**2
        score2 = res2.rvalue**2

        logger.debug(f"R-squared upper line: {score1:.3f}")
        logger.debug(f"R-squared lower line: {score2:.3f}")

        plt.plot(x1, res1.intercept + res1.slope*x1, 'b:', label=f"R-squared: {(score1+score2)/2.0:.2f}, Slope-diff: {abs(res1.slope - res2.slope):.3f}")
        plt.plot(x2, res2.intercept + res2.slope*x2, 'b:')

        plt.legend()
    plt.show()
    #plt.savefig(output_filename, dpi=300) # save image with result



if __name__ == '__main__':

    print(Path(__file__))


    args = parse_args()

    image = np.array(Image.open(args.img))
    outputdir = '.'
    i = 0

    bboxes_stitches, labels_stitches = run_stitch_detection(image, f"{outputdir}/stitch_detection_{i}.json")
    run_stitch_analyser(image, bboxes_stitches, labels_stitches, f"{outputdir}/stitch_detection_{i}.jpg")
