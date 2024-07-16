import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

import logging


logger = logging.getLogger(__name__)

try:
    from dynamic_stitch_analysis import get_subsegment_of_tracks_points
    from structure_tools import save_json, load_json
except ImportError:
    from .dynamic_stitch_analysis import get_subsegment_of_tracks_points
    from .structure_tools import save_json, load_json


def make_stitch_bboxes_global(bbox_incision, bboxes_stitches):
    return [
        [
            bbox_incision[0] + bbox_stitch[0],
            bbox_incision[1] + bbox_stitch[1],
            bbox_incision[0] + bbox_stitch[2],
            bbox_incision[1] + bbox_stitch[3]
        ]
        for bbox_stitch in bboxes_stitches
    ]



class StaticStitchAnalysis:

    def __init__(self, outputdir:Path, save_debug_images=True, show=False):
        self.outputdir = Path(outputdir)
        self.save_debug_images = save_debug_images
        self.show = show


    def pair_static_and_dynamic(self, meta:Optional[dict]=None, tool_id=1):
        """
        Run static stitch analysis.


        tool_id: int representing the tool which should be closest to the stitch center. Usually the forceps.
        """
        outputdir = self.outputdir
        if meta is None:
            meta_json_fn = outputdir / "meta.json"
            if not meta_json_fn.exists():
                with open(meta_json_fn, "r") as f:
                    meta = json.load(f)
            else:
                meta = {}

        stitch_json_fn = outputdir / "stitch_detection_0.json"
        with open(stitch_json_fn, "r") as f:
            stitch_json = json.load(f)

        bbox_incision = meta["incision_bboxes"][0]
        bboxes_stitches = stitch_json["stitch_bboxes"]
        bboxes_stitches_global = make_stitch_bboxes_global(bbox_incision, bboxes_stitches)

        # bboxes_stitches_global_centroid
        bboxes_stitches_global_centroid = np.array([
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            for bbox in bboxes_stitches_global
        ])

        self.draw_stitches(bboxes_stitches_global, bboxes_stitches_global_centroid)


        stitch_split_frames = meta["stitch_split_frames"]

        tracks_points_fn = outputdir / "tracks_points.json"

        with open(tracks_points_fn, "r") as f:
            tracks_points = json.load(f)

        tracks_points.keys()
        meta["stitch_static"] = [None] * int(len(stitch_split_frames) / 2)

        for i in range(0, len(stitch_split_frames), 2):
            start_frame = stitch_split_frames[i]
            stop_frame = stitch_split_frames[i + 1]
            tracks_points_subsegment = get_subsegment_of_tracks_points(tracks_points, start_frame, stop_frame)

            needle_holder_points_px = np.asarray(tracks_points_subsegment["data_pixels"][tool_id])
            print(needle_holder_points_px.shape)
            med = np.median(needle_holder_points_px, axis=0)


            if self.save_debug_images or self.show:
                img = self.get_img()
                if img is None:
                    return
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(img)
                plt.plot(needle_holder_points_px[:, 0], needle_holder_points_px[:, 1], "b.")
                plt.plot(med[0], med[1], "rx")
                if self.save_debug_images:
                    plt.savefig(self.outputdir / f"_static_dynamic_stitch_{i}.png")
                if self.show:
                    plt.show()

                plt.close(fig)

            # closest stitch bbox
            distances = np.linalg.norm(bboxes_stitches_global_centroid - med, axis=1)

            static_id = np.argmin(distances)

            stitch_label = stitch_json["stitch_labels"][static_id]
            stitch_id = static_id
            meta["stitch_static"][i] = {
                "static_id": stitch_id,
                "static_label": stitch_label,
                "static_bbox": bboxes_stitches_global[stitch_id]
            }

            save_json(meta, self.outputdir / f"tracks_points_stitch_{i}.json")

        return meta

    def get_img(self):
        image_fns = list(self.outputdir.glob("__cropped.*.jpg"))
        if len(image_fns) == 0:
            logger.error("First frame not found")
            return
        image_fn = image_fns[0]

        img = skimage.io.imread(image_fn)
        return img

    def draw_stitches(self, bboxes_stitches_global, bboxes_stitches_global_centroid):
        import skimage.io

        fig = plt.figure(figsize=(10, 10))
        img = self.get_img()
        if img is None:
            return

        # image_fn = outputdir / "frame_000001.png"

        plt.imshow(img)


        for i, bbox in enumerate(bboxes_stitches_global):
            plt.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]], "r")
            plt.plot(bboxes_stitches_global_centroid[:, 0], bboxes_stitches_global_centroid[:, 1], "bx")
            # show stitch number in image
            plt.text(bboxes_stitches_global_centroid[i, 0], bboxes_stitches_global_centroid[i, 1] - 20, str(i),
                     fontsize=12, color="g")
            # plt.text(bboxes_stitches_global_centroid[:, 0], bboxes_stitches_global_centroid[:, 1], "ahoje")

        if self.show:
            plt.show()
        if self.save_debug_images:
            plt.savefig(self.outputdir / "_stitches.png")

        plt.close(fig)



