import json

# from run_qr import bbox_info_extraction_from_frame
import os
import pprint
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import List, Optional, Union

import cv2
import loguru
import numpy as np
import run_qr
import tools
from incision_detection_mmdet import run_incision_detection
from loguru import logger
from media_tools import make_images_from_video
from run_perpendicular import get_frame_to_process, main_perpendicular
import datetime

# from run_mmpose import main_mmpose
from run_qr import main_qr
from run_report import bboxes_to_points, main_report

# try:
#    from .run_tracker_lite import main_tracker
#    from .run_tracker_bytetrack import main_tracker_bytetrack
#    from .run_mmpose import main_mmpose
#    from .run_qr import main_qr
#    from .run_report import main_report
#    from .run_perpendicular import main_perpendicular, get_frame_to_process
#    from .incision_detection_mmdet import run_incision_detection
# except ImportError as e:
#    logger.debug(e)
#    from run_tracker_lite import main_tracker
from run_tracker_bytetrack import main_tracker_bytetrack, main_tracker_bytetrack_batch

# from sklearn.cluster import MeanShift, estimate_bandwidth, SpectralClustering, KMeans, DBSCAN
from sklearn.cluster import KMeans
from tools import draw_bboxes_plt, save_json


# from sklearn.mixture import GaussianMixture


PROGRESS = 0
PROGRESS_MAX = 100

DEVICE = os.getenv("PIGLEG_DEVICE", default=None)
logger.debug(f"DEVICE={DEVICE}")


def set_progress(progress=None, progress_max=None):
    global PROGRESS
    global PROGRESS_MAX

    if progress:
        PROGRESS = progress
    if progress_max:
        PROGRESS_MAX = progress_max


def make_bool_from_string(s: str) -> bool:
    if type(s) == bool:
        return s
    else:
        if s.lower() in ("true", "1"):
            return True
        else:
            return False


class DoComputerVision:
    def __init__(
        self,
        filename: Path,
        outputdir: Path,
        meta: Optional[dict] = None,
        n_stitches=0,
        is_microsurgery=False,
        test_first_seconds: bool = False,
        device: Optional[str] = None,
        force_tracker:bool=False
    ):
        log_format = loguru._defaults.LOGURU_FORMAT
        self.logger_id = logger.add(
            str(Path(outputdir) / "piglegcv_log.txt"),
            format=log_format,
            level="DEBUG",
            rotation="1 week",
            backtrace=True,
            diagnose=True,
        )

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug("###############")
        logger.debug(f"device={device}")
        logger.debug(f"{test_first_seconds=}")
        logger.debug(f"{is_microsurgery=}")

        self.filename: Path = Path(filename)
        self.filename_original: Path = Path(filename)
        self.outputdir: Path = Path(outputdir)
        self.meta: dict = meta if meta is not None else {}
        self.logger_id = None
        self.frame: Optional[np.ndarray] = None
        self.filename_cropped: Optional[Path] = None
        self.test_first_seconds = test_first_seconds
        self.debug_images = {}
        self.is_microsurgery: bool = make_bool_from_string(is_microsurgery)
        self.device = device
        self.n_stitches: int = int(n_stitches)
        self.results = None
        self.force_tracker = force_tracker
        self.is_video: bool = (
            False
            if Path(self.filename).suffix.lower()
            in (
                ".png",
                ".jpg",
                ".jpeg",
                ".tiff",
                ".tif",
            )
            else True
        )

        logger.debug(f"{self.is_microsurgery=}")

    def run(self):
        self.meta = {}
        logger.info(
            f"CV processing started on {self.filename}, outputdir={self.outputdir}"
        )
        logger.debug(f"{self.is_microsurgery=}")
        logger.debug(f"{self.is_microsurgery}, {type(self.is_microsurgery)}")
        logger.debug(type(self.is_microsurgery))
        logger.debug(self.is_microsurgery)

        try:
            if Path(self.filename).suffix.lower() in (
                ".png",
                ".jpg",
                ".jpeg",
                ".tiff",
                ".tif",
            ):
                self.run_image_processing()
            else:
                # run_video_processing(filename, outputdir)
                self.run_video_processing()
            save_json(self.meta, Path(self.outputdir) / "meta.json", update=False)

            logger.debug("Work finished")
        except Exception as e:
            logger.error(traceback.format_exc())
        logger.remove(self.logger_id)

    def _make_sure_media_is_cropped(self):
        if self.filename_cropped is None:
            s = time.time()
            qr_data = self.get_parameters_for_crop_rotate_rescale()
            if qr_data["bbox_scene_area"] is not None:
                tools.draw_bbox_into_image(
                    self.frame,
                    qr_data["bbox_scene_area"],
                    linecolor=(0, 255, 0),
                    show_confidence=True,
                )
                cv2.imwrite(str(self.outputdir / "_bbox_scene_area.jpg"), self.frame)

            logger.debug(
                f"Single frame processing on original mediafile finished in {time.time() - s}s."
            )
            self.meta["duration_s_get_parameters_for_crop_rotate_rescale"] = float(
                time.time() - s
            )
            # video_preprocessing - rotate, rescale and crop -> file
            s = time.time()
            self.filename = self.do_crop_rotate_rescale(
                qr_data["bbox_scene_area"], qr_data["incision_bboxes"]
            )
            self.meta["duration_s_do_crop_rotate_rescale"] = time.time() - s
            logger.debug(f"Cropping done in {time.time() - s}s.")
            self.update_meta()

    def run_image_processing(self):
        if self.meta is None:
            self.meta = {}
        self._make_sure_media_is_cropped()
        logger.debug("Running image processing...")
        self.frame = self._get_frame_to_process_ideally_with_incision(
            self.filename_cropped, n_tries=None
        )
        # self.frame = get_frame_to_process(str(self.filename_cropped), n_tries=None)
        qr_data = run_qr.bbox_info_extraction_from_frame(
            self.frame,
            device=self.device,
            debug_image_file=self.outputdir / "_single_image_detector_results.jpg",
        )
        qr_data["qr_scissors_frames"] = []
        self.meta["qr_data"] = qr_data
        logger.debug(self.meta)

        main_perpendicular(self.frame, self.outputdir, self.meta, device=self.device)
        logger.debug("Perpendicular finished.")

    def _run_tracking(self):
        if self.is_microsurgery:
            models = [
                (
                    "./resources/tracker_model_bytetrack_microsurgery/bytetrack_pigleg.py",
                    str(
                        Path(__file__).parent
                        / "resources/tracker_model_bytetrack_microsurgery/epoch_15.pth"
                    ),
                )
            ]
            # microsurgery
            # 0: Needle holder
            # 1: Forceps
            # 2: Forceps curved
            # 3: Scissors
            class_names = {
                0: "Needle holder",
                1: "Forceps",
                2: "Forceps curved",
                3: "Scissors",
            }
        else:
            models = [
                (
                    "./resources/tracker_model_bytetrack/bytetrack_pigleg.py",
                    str(
                        Path(__file__).parent
                        / "resources/tracker_model_bytetrack/epoch.pth"
                    ),
                ),
                (
                    "./resources/tracker_model_bytetrack_hands_tools/bytetrack_pigleg.py",
                    str(
                        Path(__file__).parent
                        / "resources/tracker_model_bytetrack_hands_tools/epoch_2.pth"
                    ),
                ),
            ]
            class_names = {
                0: "Needle holder",
                1: "Forceps",
                2: "Scissors",
                10: "Needle holder bbox",
                11: "Forceps bbox",
                12: "Scissors bbox",
                13: "Left hand bbox",
                14: "Right hand bbox",
            }
        main_tracker_bytetrack(
            trackers_config_and_checkpoints=models,
            filename=self.filename,
            output_file_path=self.outputdir / "tracks.json",
            class_names=class_names,
            device=self.device,
            additional_hash="s" if self.test_first_seconds else "f",
            force_tracker=self.force_tracker
        )

    def run_video_processing(self):

        """

        :param filename:
        :param outputdir:
        :param meta: might be used for progressbar
        :return:
        """
        logger.debug("Running video processing...")
        if self.meta is None:
            self.meta = {}

        set_progress(1)
        # get_sigle_frame
        # single_frame_processing ->
        # video_preprocessing - rotate, rescale and crop -> file
        # single_frame_processing on rotated
        # bytrack
        # make_report

        self._make_sure_media_is_cropped()

        s = time.time()
        self.run_image_processing()
        logger.debug(
            f"Single frame processing on cropped mediafile finished in {time.time() - s}s."
        )
        self.meta["duration_s_run_image_processing"] = float(time.time() - s)
        logger.debug(f"Image processing finished in {time.time() - s}s.")

        self.meta["is_microsurgery"] = self.is_microsurgery
        self.meta["n_stitches_by_user"] = self.n_stitches

        s = time.time()
        self._run_tracking()
        self.meta["duration_s_tracking"] = float(time.time() - s)
        logger.debug(f"Tracker finished in {time.time() - s}s.")
        set_progress(50)

        logger.debug(f"filename={self.filename}, outputdir={self.outputdir}")
        logger.debug(
            f"filename={Path(self.filename).exists()}, outputdir={Path(self.outputdir).exists()}"
        )

        s = time.time()
        self._find_stitch_ends_in_tracks(
            n_clusters=self.n_stitches,
            plot_clusters=True,
            clusters_image_path=self.outputdir / "_stitch_clusters.jpg",
        )
        self.meta["duration_s_stitch_ends"] = float(time.time() - s)
        logger.debug(f"{self.meta['stitch_split_frames']=}")
        logger.debug(f"Stitch ends found in {time.time() - s}s.")

        s = time.time()
        self._make_report(cut_frames=self.meta["stitch_split_frames"])
        set_progress(70)
        if "stitch_scores" in self.meta:
            if len(self.meta["stitch_scores"]) > 0:
                self.results["Stitches linearity score [%]"] = self.meta["stitch_scores"][0]["r_score"] * 100
                self.results["Stitches parallelism score [%]"] = self.meta["stitch_scores"][0]["s_score"] * 100
                self.results["Stitches perpendicular score [%]"] = self.meta["stitch_scores"][0]["p_score"] * 100
        # save statistic to file
        
        self.meta["duration_s_report"] = float(time.time() - s)
        self.meta["processed_at"]=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_results()

        set_progress(99)

        logger.debug(f"Report finished in {time.time() - s}s.")

        logger.debug("Report based on video is finished.")
        logger.debug("Video processing finished")

    def _get_frame_to_process_ideally_with_incision(
        self,
        filename,
        return_qrdata=False,
        n_tries=None,
        debug_image_file: Optional[Path] = None,
    ):
        if self.is_video:
            frame_from_end = 0
            for i in range(10):
                frame, local_meta = get_frame_to_process(
                    str(filename),
                    n_tries=n_tries,
                    return_metadata=True,
                    reference_frame_position_from_end=frame_from_end,
                )
                qr_data = run_qr.bbox_info_extraction_from_frame(
                    frame, device=self.device, debug_image_file=debug_image_file
                )
                if len(qr_data["incision_bboxes"]) > 0:
                    logger.debug(
                        f"Found incision bbox in frame {frame_from_end} from the end."
                    )
                    break
                else:
                    frame_from_end = (
                        local_meta["reference_frame_position_from_end"] + 10
                    )
            logger.debug(
                f"Incision bbox not found. Using in frame {frame_from_end} frame from the end."
            )
        else:
            frame, _ = get_frame_to_process(
                str(filename),
                n_tries=n_tries,
                return_metadata=True,
            )
            qr_data = run_qr.bbox_info_extraction_from_frame(frame, device=self.device)
        if return_qrdata:
            return frame, qr_data
        else:
            return frame

    def get_parameters_for_crop_rotate_rescale(self):
        logger.debug(f"device={self.device}")
        if self.is_video:
            self.frame, qr_data = self._get_frame_to_process_ideally_with_incision(
                self.filename_original,
                return_qrdata=True,
                debug_image_file=self.outputdir
                / "_single_image_detector_results_full_size.jpg",
            )
        else:
            frame, local_meta = get_frame_to_process(self.filename_original)
        qr_data["qr_scissors_frames"] = []
        imgs, bboxes = run_incision_detection(self.frame, device=self.device)
        qr_data["incision_bboxes_old"] = bboxes.tolist()
        #         print(qr_data)
        #         fig = draw_bboxes(self.frame[:,:,::-1], qr_data["incision_bboxes"])
        #         fig = draw_bboxes(self.frame[:,:,::-1], qr_data["bbox_scene_area"])
        #         from matplotlib import pyplot as plt
        #         plt.show()
        #         self.debug_images["crop_rotate_rescale_parameters":img]
        logger.debug(f"{qr_data=}")
        return qr_data

    def do_crop_rotate_rescale(
        self,
        crop_bbox: Optional[list] = None,
        incision_bboxes: Optional[list] = None,
        crop_bbox_score_threshold=0.5,
    ) -> Path:

        # base_name, extension = str(self.filename).rsplit('.', 1)

        transpose = False
        if not self.is_microsurgery:
            if incision_bboxes is not None and len(incision_bboxes) > 0:
                width = int(incision_bboxes[0][2] - incision_bboxes[0][0])
                height = int(incision_bboxes[0][3] - incision_bboxes[0][1])

                if width > height:
                    transpose = False
                else:
                    transpose = True
            elif self.frame.shape[0] > self.frame.shape[1]:
                transpose = True
        self.filename_cropped = self.outputdir / "__cropped.mp4"

        # Recreate the modified file path
        # new_file_path = new_base_name + '.' + "mp4"
        logger.debug(f"self.filename_cropped={self.filename_cropped}")

        # s = ["ffmpeg", '-i', str(self.filename), '-ac', '2', "-y", "-b:v", "2000k", "-c:a", "aac", "-c:v", "libx264", "-b:a", "160k",
        #      "-vprofile", "high", "-bf", "0", "-strict", "experimental", "-f", "mp4", base_name]

        meta = self.meta
        # get FPS, frame_count and frame size from original video
        cap = cv2.VideoCapture(str(self.filename))
        self.meta["orig fps"] = int(cap.get(cv2.CAP_PROP_FPS))
        self.meta["orig totalframecount"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta["orig frame_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.meta["orig frame_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        filter_str = ""

        if crop_bbox is not None:
            if crop_bbox[4] > crop_bbox_score_threshold:
                cr_out_w = int(crop_bbox[2] - crop_bbox[0])
                cr_out_h = int(crop_bbox[3] - crop_bbox[1])
                cr_x = int(crop_bbox[0])
                cr_y = int(crop_bbox[1])
                filter_str += f"crop={cr_out_w}:{cr_out_h}:{cr_x}:{cr_y},"
        if transpose:
            filter_str += "transpose=1,"

        orig_fps = self.meta["orig fps"]
        # if orig_fps >= 30:
        #     new_fps= orig_fps / 2
        #     filter_str += f"fps=fps={new_fps:0.2f},"
        #     filter
        # elif orig_fps >= 60:
        #     new_fps = orig_fps / 4
        #     filter_str += f"fps=fps={new_fps:0.2f},"
        # elif orig_fps >= 120:
        #     new_fps = orig_fps / 8
        #     filter_str += f"fps=fps={new_fps:0.2f},"
        filter_str += f"fps=fps=15,"

        filter_str += "scale=720:trunc(ow/a/2)*2"

        additional_params = []

        if self.test_first_seconds:
            additional_params.extend(["-t", "5"])

        logger.debug(f"filename={self.filename}, {self.filename.exists()}")
        s = (
            ["ffmpeg", "-i", str(self.filename)]
            + additional_params
            + [
                "-filter:v",
                filter_str,
                "-an",
                "-y",
                "-b:v",
                "1000k",
                str(self.filename_cropped),
            ]
        )
        logger.debug(f"{' '.join(s)}")
        try:
            subprocess.check_output(s, shell=False, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )

        logger.debug(
            f"filename_cropped={self.filename_cropped}, {self.filename_cropped.exists()}"
        )
        make_images_from_video(
            self.filename_cropped,
            outputdir=self.outputdir,
            filemask=str(
                self.filename_cropped.with_suffix(self.filename_cropped.suffix + ".jpg")
            ),
            n_frames=1,
            create_meta_json=False,
        )
        return self.filename_cropped
        # return self.filename

    def update_meta(self, filename=None):
        if filename is None:
            filename = self.filename
        cap = cv2.VideoCapture(str(filename))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.meta.update(
            {"filename_full": str(filename), "fps": fps, "frame_count": totalframecount}
        )

    def _find_stitch_ends_in_tracks(
        self,
        n_clusters: int,
        # tool_index: int = 1,
        weight_of_later=0.9,
        plot_clusters=False,
        clusters_image_path: Optional[Path] = None,
        # trim_tool_index:int = 0
    ) -> List:
        self.meta["stitch_split_frames"] = []
        self.meta["stitch_split_s"] = []

        if self.is_microsurgery:
            tool_index = [1,2]
            trim_tool_index = [0,1,2]
        else:
            tool_index = [1]
            trim_tool_index = [0,1]

        try:
            n_clusters = int(n_clusters)

            # this will create "tracks_points.json" and it is called in the processing twice. The second call is later in self._make_report()
            bboxes_to_points(str(self.outputdir))
            # split_frames = []
            split_s, split_frames = find_stitch_ends_in_tracks(
                self.outputdir,
                n_clusters=n_clusters,
                tool_indexes=tool_index,
                weight_of_later=weight_of_later,
                metadata=self.meta,
                plot_clusters=plot_clusters,
                clusters_image_path=clusters_image_path,
                trim_tool_indexes=trim_tool_index
            )
            # self.meta["qr_data"]["stitch_split_frames"] = split_frames
            self.meta["stitch_split_frames"] = split_frames
            self.meta["stitch_split_s"] = split_s
            return split_frames
        except Exception as e:
            logger.error(f"Error in find_stitch_ends_in_tracks: {e}")
            print(traceback.format_exc())
        return []

    def _make_report(self, cut_frames=[]):
        #                            cut_frames=self.meta["stitch_split_frames"]
        #                            )
        self.results = main_report(
            self.filename,
            self.outputdir,
            meta=self.meta,
            is_microsurgery=self.is_microsurgery,
            cut_frames=cut_frames,
            test_first_seconds=self.test_first_seconds,
        )
        return self.results

    def _save_results(self):
        logger.debug("Saving results...")
        save_json(self.results, self.outputdir / "results.json")

    def _load_meta(self):
        # if metadata is None:
        meta_path = self.outputdir / "meta.json"
        assert meta_path.exists()
        with open(meta_path, "r") as f:
            self.meta = json.load(f)


def do_computer_vision(
    filename,
    outputdir,
    meta=None,
    is_microsurgery: bool = False,
    n_stitches: Optional[int] = None,
    device=DEVICE,
    force_tracker:bool = False
):
    logger.debug(f"{is_microsurgery=}")
    return DoComputerVision(
        filename,
        outputdir,
        meta,
        is_microsurgery=is_microsurgery,
        n_stitches=n_stitches,
        device=device,
        force_tracker=force_tracker
    ).run()


def _get_X_px_fr_more_tools(data: dict, incision_bboxes: list, tool_indexes:List[int], time_axis=2) -> np.ndarray:
    # merge several tools
    X_px_fr_list = []
    for trim_tool_index in tool_indexes:
        X_px_fr_list.append(_get_X_px_fr(data, incision_bboxes, trim_tool_index))

    # merge the lists
    X_px_fr = np.concatenate(X_px_fr_list, axis=0)
    #sort by time_axis
    X_px_fr = X_px_fr[X_px_fr[:, time_axis].argsort()]
    return X_px_fr


def _get_X_px_fr(data:dict, incision_bboxes:list, tool_index:int) -> np.ndarray:
    """Get X vector in pixels and frames filtered to the incision area."""
    if "data_pixels" in data:
        X_px = np.asarray(data["data_pixels"][tool_index])
    else:
        # backward compatibility
        X_px = np.asarray(data[f"data_pixels_{tool_index}"])

    time_fr = np.asarray(data["frame_ids"][tool_index]).reshape(-1, 1)
    X_px_fr = np.concatenate([X_px, time_fr], axis=1)

    if len(incision_bboxes) > 0:

        X_px_fr_tmp = tools.filter_points_in_bbox(
            X_px_fr, tools.make_bbox_larger(incision_bboxes[0], 2.0)
        )
        logger.debug(f"{X_px_fr.shape=}, {X_px_fr_tmp.shape=}")
        X_px_fr = X_px_fr_tmp

    return X_px_fr


def find_stitch_ends_in_tracks(
    outputdir,
    n_clusters: int,
    tool_indexes:Union[int,List[int]] = 1,
    trim_tool_indexes:Union[int,List[int]] = 0,
    weight_of_later=0.9,
    metadata=None,
    plot_clusters=False,
    clusters_image_path: Optional[Path] = None,
):
    """
    Find stitch ends in tracks.
    :param outputdir:
    :param n_clusters: if zero or one, no clustering is done, just trimming the video
    :param tool_indexes: tool used for splitting video to the parts
    :param weight_of_later:
    :param metadata:
    :param plot_clusters:
    :param clusters_image_path:
    :param trim_tool_indexes: tool used for trimming of the video parts

    """
    time_axis: int = 2
    logger.debug(f"find_stitch_end, {n_clusters=}, {outputdir=}")

    if type(tool_indexes) == int:
        tool_indexes = [tool_indexes]
    if type(trim_tool_indexes) == int:
        trim_tool_indexes = [trim_tool_indexes]

    points_path = outputdir / "tracks_points.json"
    assert points_path.exists()
    time_axis = int(time_axis)
    with open(points_path, "r") as f:
        data = json.load(f)

    if metadata is None:
        meta_path = outputdir / "meta.json"
        assert points_path.exists()
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    incision_bboxes = []
    if ("incision_bboxes" in metadata) and (len(metadata["incision_bboxes"]) > 0):
        incision_bboxes = metadata["incision_bboxes"]
    elif "incision_bboxes" in metadata["qr_data"]:
        incision_bboxes = metadata["qr_data"]["incision_bboxes"]
    logger.debug(f"{incision_bboxes=}")

    logger.debug(
        f"find stitch end, pix_size={metadata['qr_data']['pix_size']}, fps={metadata['fps']}"
    )

    X_px_fr = _get_X_px_fr_more_tools(data, incision_bboxes, tool_indexes, time_axis=time_axis)
    # pix_size is in [m] to normaliza data a bit we use [mm]
    axis_normalization = np.asarray(
        [
            metadata["qr_data"]["pix_size"] * 1000,
            metadata["qr_data"]["pix_size"] * 1000,
            1.0 / metadata["fps"],
        ]
    )

    X = X_px_fr * axis_normalization

    if n_clusters > 1:

        # time =  np.asarray(list(range(X.shape[0]))).reshape(-1,1)
        # time =  np.asarray(data["frame_ids"][tool_index]).reshape(-1,1) / metadata["fps"]

        # X = np.concatenate([X, time], axis=1)
        # X = X * axis_normalization

        # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
        # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

        ms = KMeans(n_clusters=n_clusters)
        # ms = DBSCAN()
        # ms = SpectralClustering()
        # ms = SpectralClustering(3, affinity='precomputed', n_init=100,
        #                       assign_labels='discretize')
        # ms = GaussianMixture()
        ms.fit(X)
        labels = ms.labels_
        # labels = ms.predict(X)
        cluster_centers = ms.cluster_centers_

        prev = labels[0]
        splits_s = []
        splits_frames = []
        for frame_i, label in enumerate(labels):
            if label != prev:
                time = ((1 - weight_of_later) * X[frame_i - 1, time_axis]) + (
                    weight_of_later * X[frame_i, time_axis]
                )
                splits_s.append(time)
                splits_frames.append(int(time * float(metadata["fps"])))
            prev = label
    else:
        # splits_s = [np.mean(X[:, time_axis])]
        # splits_frames = [int(splits_s[0] * float(metadata["fps"]))]
        splits_s = []
        splits_frames = []
        cluster_centers = [np.mean(X, axis=0)]
        labels = np.zeros(X.shape[0])


    # print(f"{splits_s=}")
    # print(f"{splits_frames=}")

    X_px_fr = _get_X_px_fr_more_tools(data, incision_bboxes, trim_tool_indexes, time_axis=time_axis)

    actual_split_i = 0
    key_frame = int(X_px_fr[0][time_axis])
    new_splits_frames = [key_frame]
    new_splits_s = [float(key_frame) / float(metadata["fps"])]
    logger.debug(f"{splits_frames=}")
    new_labels = []
    
    if len(splits_frames) > 0:
        for i in range(1, X_px_fr.shape[0]):
            X_px_fr_i_prev = X_px_fr[i-1]
            X_px_fr_i = X_px_fr[i]
            # new_labels.append(actual_split_i)
    #         print(f"{X_px_fr_i=}")

    #         print(f"{actual_split_i=}")


            if X_px_fr_i[time_axis] > splits_frames[actual_split_i]:
                # end of previous split
                new_splits_frames.append(int(X_px_fr_i_prev[time_axis]))
                new_splits_s.append(float(X_px_fr_i_prev[time_axis]) / float(metadata["fps"]))

                # start of next split
                new_splits_frames.append(int(X_px_fr_i[time_axis]))
                new_splits_s.append(float(X_px_fr_i[time_axis]) / float(metadata["fps"]))
                actual_split_i += 1
                if actual_split_i >= len(splits_frames):
                    break
                

    # end of last split
    new_splits_frames.append(int(X_px_fr[-1][time_axis]))
    new_splits_s.append(float(X_px_fr[-1][time_axis]) / float(metadata["fps"]))


        # if X_px_fr_i[time_axis] > splits_s[-1]:
        #     new_splits_s.append(X_px_fr_i[time_axis])
        #     new_splits_frames.append(int(X_px_fr_i[time_axis] * float(metadata["fps"])))
        #     break
        

    # new_split_s = []
        
        
        
    if plot_clusters:
        plot_track_clusters(
            X,
            labels,
            cluster_centers,
            new_splits_s,
            clusters_image_path=clusters_image_path,
        )


    return new_splits_s, new_splits_frames
    # return splits_s, splits_frames


def plot_track_clusters(
    X, labels, cluster_centers, splits_s, clusters_image_path: Optional[Path] = None
):
    from matplotlib import pyplot as plt

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    fig = plt.figure()
    # plt.subplot(211)
    plt.subplot(121)
    # plt.clf()

    colors = [
        "#dede00",
        "#377eb8",
        "#f781bf",
        "#81bf37",
        "#bf3781",
        "#f3781b",
        "#eb88b1",
        "#1bff78",
    ]
    markers = [ "x", "o","^","s","x","o","^","x","o","^","x","o","^"]
    markers_x = ["x" for i in range(len(colors))]
    markers_o = ["x" for i in range(len(colors))]
    
    ax0 = 1
    ax1 = 0

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, ax0], X[my_members, ax1], markers_x[k], color=col)
        plt.plot(
            cluster_center[ax0],
            cluster_center[ax1],
            markers[k],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
    plt.xlabel("[mm]")
    plt.ylabel("[mm]")
    # plt.title("Estimated number of clusters: %d" % n_clusters_)

    # plt.subplot(212)
    plt.subplot(122)
    colors = [
        "#dede00",
        "#377eb8",
        "#f781bf",
        "#81bf37",
        "#bf3781",
        "#f3781b",
        "#eb88b1",
        "#1bff78",
    ]


    ax0 = 2
    ax1 = 0
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, ax0], X[my_members, ax1], markers_x[k], color=col)
        plt.plot(
            cluster_center[ax0],
            cluster_center[ax1],
            markers[k],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
    plt.xlabel("[s]")
    colors = ["g", 'r']
    for i, yline in enumerate(splits_s):
        # if odd use green, if even use red
        plt.axvline(x=yline, c=colors[i%2])
        # plt.axhline(y=yline, c=colors[i%2])


    if clusters_image_path is not None:
        plt.savefig(clusters_image_path)
        plt.close(fig)


# def do_computer_vision_2(filename, outputdir, meta):
#     log_format = loguru._defaults.LOGURU_FORMAT
#     logger_id = logger.add(
#         str(Path(outputdir) / "piglegcv_log.txt"),
#         format=log_format,
#         level="DEBUG",
#         rotation="1 week",
#         backtrace=True,
#         diagnose=True,
#     )
#     logger.debug(f"CV processing started on {filename}, outputdir={outputdir}")
#
#     try:
#         if Path(filename).suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
#             run_image_processing(filename, outputdir)
#         else:
#             #run_video_processing(filename, outputdir)
#             run_video_processing2(filename, outputdir)
#
#         logger.debug("Work finished")
#     except Exception as e:
#         logger.error(traceback.format_exc())
#     logger.remove(logger_id)


# def run_video_processing(filename: Path, outputdir: Path) -> dict:
#     """
#     Deprecated
#     :param filename:
#     :param outputdir:
#     :return:
#     """
#     logger.debug("Running video processing...")
#     s = time.time()
#     main_qr(filename, outputdir)
#     logger.debug(f"QR finished in {time.time() - s}s.")
#
#     s = time.time()
#     tracker_model_path = Path(__file__).parent / "./.cache/tracker_model"
#     if not tracker_model_path.exists():
#         tracker_model_path = Path(__file__).parent / "resources/tracker_model"
#     main_tracker("{} \"{}\" --output_dir {}".format(tracker_model_path, filename, outputdir))
#     logger.debug(f"Tracker finished in {time.time() - s}s.")
#
#     #
#     # s = time.time()
#     # main_mmpose(filename, outputdir)
#     # logger.debug(f"MMpose finished in {time.time() - s}s.")
#
#
#     main_report(filename, outputdir)
#     logger.debug("Report based on video is finished.")
#
#     # if extention in images_types:
#
#     run_image_processing(filename, outputdir)
#     # logger.debug("Perpendicular finished.")
#     logger.debug("Video processing finished")

# def run_video_processing2(filename: Path, outputdir: Path, meta:dict=None) -> dict:
#     """
#
#     :param filename:
#     :param outputdir:
#     :param meta: might be used for progressbar
#     :return:
#     """
#     logger.debug("Running video processing...")
#     if meta is None:
#         meta = {}
#     s = time.time()
#     main_qr(filename, outputdir)
#     logger.debug(f"QR finished in {time.time() - s}s.")
#     run_image_processing(filename, outputdir, skip_qr=True)
#     s = time.time()
#     logger.debug(f"Image processing finished in {time.time() - s}s.")
#
#     # main_tracker_bytetrack("\"{}\" \"{}\" \"{}\" --output_dir \"{}\"".format('./resources/tracker_model_bytetrack/bytetrack_pigleg.py','./resources/tracker_model_bytetrack/epoch.pth', filename, outputdir))
#     # f"\"./resources/tracker_model_bytetrack/bytetrack_pigleg.py\" \"{filename}\" --output_dir \"{outputdir}\"",
#     main_tracker_bytetrack(
#         config_file="./resources/tracker_model_bytetrack/bytetrack_pigleg.py",
#         filename=filename,
#         output_dir=outputdir,
#         checkpoint=Path(__file__).parent / "resources/tracker_model_bytetrack/epoch.pth",
#         device="cuda"
#     )
#     # run_media_processing(Path(filename), Path(outputdir))
#     logger.debug(f"Tracker finished in {time.time() - s}s.")
#
#     #
#     # s = time.time()
#     # main_mmpose(filename, outputdir)
#     # logger.debug(f"MMpose finished in {time.time() - s}s.")
#
#     logger.debug(f"filename={filename}, outputdir={outputdir}")
#     logger.debug(f"filename={Path(filename).exists()}, outputdir={Path(outputdir).exists()}")
#
#     main_report(filename, outputdir)
#
#     logger.debug("Report based on video is finished.")
#
#     # if extention in images_types:
#
#     # logger.debug("Perpendicular finished.")
#     logger.debug("Video processing finished")


# def run_image_processing(filename: Path, outputdir: Path, skip_qr=False, device="cpu") -> dict:
#     logger.debug("Running image processing...")
#     frame = get_frame_to_process(str(filename))
#     run_qr.bbox_info_extraction_from_frame(frame)
#     main_perpendicular(filename, outputdir, device=device)
#     logger.debug("Perpendicular finished.")
#     # TODO add predict image
#     # img = mmcv.imread(str(img_fn))
#     # run_incision_detection(filename, outputdir)


def _make_images_from_video(filename: Path, outputdir: Path) -> Path:
    outputdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(filename))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()

        frame_id += 1
        if not ret:
            break
        else:
            file_name = "{}/frame_{:0>6}.png".format(outputdir, frame_id)
            cv2.imwrite(file_name, frame)
            logger.trace(file_name)
    cap.release()

    metadata = {
        "filename_full": str(filename),
        "fps": fps,
        "frame_count": totalframecount,
    }
    json_file = outputdir / "meta.json"
    save_json(metadata, json_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process media file")
    parser.add_argument("filename", type=str)
    parser.add_argument("outputdir", type=str)
    args = parser.parse_args()
    do_computer_vision(Path(args.filename), Path(args.outputdir))
