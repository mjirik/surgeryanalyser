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
from incision_detection_mmdet import run_incision_detection
from loguru import logger
from media_tools import make_images_from_video
from run_perpendicular import get_frame_to_process, main_perpendicular
import datetime
import sklearn

# from run_mmpose import main_mmpose
from run_qr import main_qr
from run_report import convert_track_bboxes_to_center_points, main_report

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

try:
    import tools
    from tools import save_json
except ImportError:
    from .tools import save_json
    from . import tools


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
        self.frame_at_beginning: Optional[np.ndarray] = None
        self.filename_cropped: Optional[Path] = None
        self.test_first_seconds = test_first_seconds
        self.debug_images = {}
        self.is_microsurgery: bool = make_bool_from_string(is_microsurgery)
        self.device = device
        self.n_stitches: int = int(n_stitches)
        self.results = None
        self.force_tracker = force_tracker
        self.operating_area_bbox = None
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

    # def _find_frame_from_start_of_video(self):
    #
    #     logger.debug("Looking from the frame from the video start...")
    #     self.frame = self._get_frame_to_process_ideally_with_incision(
    #         self.filename_cropped, n_tries=None
    #     )
    #     # self.frame = get_frame_to_process(str(self.filename_cropped), n_tries=None)
    #     qr_data = run_qr.bbox_info_extraction_from_frame(
    #         self.frame,
    #         device=self.device,
    #         debug_image_file=self.outputdir / "_single_image_detector_results.jpg",
    #     )

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

        main_perpendicular(self.frame, self.outputdir, self.meta, device=self.device, img_alternative=self.frame_at_beginning)
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
                        / "resources/tracker_model_bytetrack_hands_tools/epoch.pth"
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

    def prepare_operation_area_bbox(self):
        _, points_per_tool, _ = convert_track_bboxes_to_center_points(str(self.outputdir))
        points_all_tools = np.concatenate(points_per_tool, axis=0)
        from .run_report import find_incision_bbox_with_highest_activity
        incision_bboxes = self.meta["qr_data"]["incision_bboxes"]
        oa_bbox = find_incision_bbox_with_highest_activity(points_all_tools, incision_bboxes)
        # make bbox larger
        oa_bbox = tools.make_bbox_larger(oa_bbox, 2.0)
        self.operating_area_bbox = find_incision_bbox_with_highest_activity(points_all_tools, incision_bboxes)
        return self.operating_area_bbox

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
        logger.debug("Searching for frame at beginning")
        self.frame_at_beginning = self._get_frame_to_process_ideally_with_incision(
            self.filename_cropped,
            n_tries=100,
            frame_from_end=-1,
            frame_from_end_step=-5
        )
        logger.debug(f"{self.frame_at_beginning=}, {type(self.frame_at_beginning)=}")
        if self.frame_at_beginning is not None:
            cv2.imwrite(str(self.outputdir / "_frame_at_beginning_with_incision.jpg"), self.frame_at_beginning)
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
        oa_bbox = self.prepare_operation_area_bbox()

        s = time.time()

        stitch_split_frames = self._find_stitch_ends_in_annotation()

        if len(stitch_split_frames) == 0:
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
        frame_from_end_step: int = 5,
        n_detection_tries: int = 180,
        frame_from_end: int = 0,
    ):
        # FPS=15, n_detection_tries * frame_from_end_step = 450 => cca 60 sec.
        # FPS=30, n_detection_tries * frame_from_end_step = 450 => cca 30 sec.
        # remember at least some frame for the case that no incision is found and we will run out of frames
        bad_last_frame = None
        bad_qr_data = None
        if self.is_video:
            # frame_from_end = 0
            for i in range(n_detection_tries):
                frame, local_meta = get_frame_to_process(
                    str(filename),
                    n_tries=n_tries,
                    return_metadata=True,
                    reference_frame_position_from_end=frame_from_end,
                    step=frame_from_end_step
                )
                if frame is None:
                    logger.debug("Frame is None.")
                    frame = bad_last_frame
                    qr_data = bad_qr_data
                    break
                else:
                    try:
                        qr_data = run_qr.bbox_info_extraction_from_frame(
                            frame, device=self.device, debug_image_file=debug_image_file
                        )
                    except IndexError as e:
                        logger.error(f"Error in bbox_info_extraction_from_frame: {e}")
                        logger.error(traceback.format_exc())
                        logger.debug(f"{type(frame)=}")
                        logger.debug(f"{frame.shape=}")
                        qr_data = bad_qr_data
                        frame = bad_last_frame
                        break
                    bad_last_frame = frame
                    bad_qr_data = qr_data
                    if len(qr_data["incision_bboxes"]) > 0:
                        logger.debug(
                            f"Found incision bbox in frame {frame_from_end} from the end."
                        )
                        break
                    else:
                        frame_from_end = (
                            local_meta["reference_frame_position_from_end"] + frame_from_end_step
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
            self.frame, local_meta = get_frame_to_process(self.filename_original)
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
        ffmpeg_subprocess_params = ' '.join(s)
        logger.debug(f"{' '.join(s)}")
        self.meta["ffmpeg_subprocess_params"] = ffmpeg_subprocess_params
        prev_meta = self._check_meta_and_load()

        try:
            if prev_meta and ("ffmpeg_subprocess_params" in prev_meta) and (
                self.meta["ffmpeg_subprocess_params"] == prev_meta["ffmpeg_subprocess_params"]
            ):
                logger.debug("Skipping video resize. It was done before.")
            else:
                subprocess.check_output(s, shell=False, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )

        logger.debug(
            f"{self.filename_cropped=}, {self.filename_cropped.exists()=}"
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
            {
                "filename_full": str(filename),
                "fps": fps,
                "frame_count": totalframecount,
                "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
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

        oa_bbox=self.operating_area_bbox

        if self.is_microsurgery:
            tool_index = [0,1,2]
            trim_tool_index = [0,1,2]
        else:
            tool_index = [0,1]
            trim_tool_index = [0,1]

        try:
            n_clusters = int(n_clusters)

            # this will create "tracks_points.json" and it is called in the processing twice. The second call is later in self._make_report()
            # split_frames = []
            split_s, split_frames = find_stitch_ends_in_tracks(
                self.outputdir,
                n_clusters=n_clusters,
                tool_indexes=tool_index,
                weight_of_later=weight_of_later,
                metadata=self.meta,
                plot_clusters=plot_clusters,
                clusters_image_path=clusters_image_path,
                trim_tool_indexes=trim_tool_index,
                oa_bbox=oa_bbox
            )
            # self.meta["qr_data"]["stitch_split_frames"] = split_frames
            self.meta["stitch_split_frames"] = split_frames
            self.meta["stitch_split_s"] = split_s
            return split_frames
        except Exception as e:
            logger.error(f"Error in find_stitch_ends_in_tracks: {e}")
            logger.error(traceback.format_exc())
            print(traceback.format_exc())
        return []

    def _find_stitch_ends_in_annotation(self) -> List:
        stitch_split_s = []
        fn = self.outputdir / "annotation_0.json"
        if fn.exists():
            with open(fn, "r") as f:
                data = json.load(f)

        if "annotation" in data:
            text_annotation = data["annotation"]
            # typically the annotation looks like:
            # 00:00:40 stitch_start
            # 00:02:10 stitch_end
            # 00:02:24 stitch_start
            # 00:03:45 stitch_end

            # find all stitch_start and stitch_end
            stitch_events = []
            for line in text_annotation.split("\n"):
                if "stitch_start" in line or "stitch_end" in line:
                    # parse time

                    hours, minutes, seconds = line.split(" ")[0].split(":")
                    seconds_total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                    rest_of_line = line.split(" ")[1:]

                    stitch_events.append([seconds_total, rest_of_line])

            # sort by time
            stitch_events = sorted(stitch_events, key=lambda x: x[0])
            logger.debug(f"{stitch_events=}")

            # take just the time
            stitch_split_s = [x[0] for x in stitch_events]
        fps = self.meta["fps"]
        # convert to frames
        stitch_split_frames = [int(x * fps) for x in stitch_split_s]


        # TODO implement
        self.meta["stitch_split_frames"] = stitch_split_frames
        self.meta["stitch_split_s"] = stitch_split_s
        return stitch_split_frames

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
            oa_bbox=self.operating_area_bbox
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

    def _check_meta_and_load(self) -> Optional[dict]:
        meta_path = self.outputdir / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r") as f:
            meta = json.load(f)

        return meta



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


def add_dim_with_cumulative_number_of_empty_frames(X_px_fr: np.ndarray, empty_frames_axis:int=3) -> np.ndarray:
    """
    Add dimension with cumulative number of empty frames.
    :param X_px_fr: X_px_fr
    :return: X_px_fr with added dimension
    """
    # add dimension with cumulative number of empty frames
    X_px_fr = np.concatenate(
        [X_px_fr, np.zeros((X_px_fr.shape[0], 1), dtype=X_px_fr.dtype)], axis=1
    )
    empty_frames = 0

    for i in range(1, X_px_fr.shape[0]):
        if X_px_fr[i, 2] - X_px_fr[i - 1, 2] > 1:
            empty_frames += (X_px_fr[i, 2] - X_px_fr[i - 1, 2]) - 1
        X_px_fr[i, empty_frames_axis] = empty_frames

    logger.debug(f"{X_px_fr.shape[0]=}")
    logger.debug(f"{empty_frames=}")
    logger.debug(f"{X_px_fr[-1, 2]=}")
    return X_px_fr



def _get_X_px_fr_more_tools(data: dict, oa_bbox: Union[list, None], tool_indexes:List[int], time_axis=2) -> np.ndarray:
    # merge several tools
    X_px_fr_list = []
    for trim_tool_index in tool_indexes:
        X_px_fr_list.append(_get_X_px_fr(data, oa_bbox, trim_tool_index))

    logger.debug(f"{X_px_fr_list=}")
    # merge the lists
    X_px_fr = np.concatenate(X_px_fr_list, axis=0)
    logger.debug(f"{X_px_fr.shape=}")
    #sort by time_axis
    X_px_fr = X_px_fr[X_px_fr[:, time_axis].argsort()]
    return X_px_fr


def _get_X_px_fr(data:dict, oa_bbox:Optional[list], tool_index:int) -> np.ndarray:
    """Get X vector in pixels and frames filtered to the incision area."""
    if "data_pixels" in data:
        X_px = np.asarray(data["data_pixels"][tool_index])
    else:
        # backward compatibility
        X_px = np.asarray(data[f"data_pixels_{tool_index}"])

    time_fr = np.asarray(data["frame_ids"][tool_index]).reshape(-1, 1)
    X_px_fr = np.concatenate([X_px, time_fr], axis=1)

    if oa_bbox is not None:

        # TODO where is the funciton located?
        X_px_fr_tmp = tools.filter_points_in_bbox(
            # X_px_fr, tools.make_bbox_larger(incision_bbox, 2.0)
            X_px_fr, oa_bbox
        )
        logger.debug(f"{X_px_fr.shape=}, {X_px_fr_tmp.shape=}")
        X_px_fr = X_px_fr_tmp

    return X_px_fr



def _get_metadata(outputdir, metadata=None):


    points_path = outputdir / "tracks_points.json"
    assert points_path.exists()

    with open(points_path, "r") as f:
        track_points = json.load(f)

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

    return track_points, metadata, incision_bboxes

def _smooth_in_1D(X, labels, time_axis=2, n_neighbors=100):
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X[:,time_axis].reshape(-1,1), labels)
    labels1 = clf.predict(X[:,time_axis].reshape(-1,1))
    return labels1


def _get_splits(X, labels, fps, time_axis=2, weight_of_later=0.6):
    prev = labels[0]
    splits_s = []
    splits_frames = []
    for frame_i, label in enumerate(labels):
        if label != prev:
            time = ((1 - weight_of_later) * X[frame_i - 1, time_axis]) + (
                weight_of_later * X[frame_i, time_axis]
            )
            splits_s.append(time)
            splits_frames.append(int(time * float(fps)))
        prev = label

    return splits_s, splits_frames


def find_stitch_ends_in_tracks(
    outputdir,
    n_clusters: int,
    tool_indexes:Union[int,List[int]] = [0, 1],
    trim_tool_indexes:Union[int,List[int]] = [0,1],
    weight_of_later=0.6,
    metadata=None,
    plot_clusters=False,
    clusters_image_path: Optional[Path] = None,
    oa_bbox=None
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
    empty_frame_axis:int = 3

    if type(tool_indexes) == int:
        tool_indexes = [tool_indexes]
    if type(trim_tool_indexes) == int:
        trim_tool_indexes = [trim_tool_indexes]

    logger.debug(f"find_stitch_end, {n_clusters=}, {outputdir=}")
    data, metadata, _ = _get_metadata(outputdir, metadata)
    X_px_fr = _get_X_px_fr_more_tools(data, oa_bbox, tool_indexes, time_axis=time_axis)
    # pix_size is in [m] to normaliza data a bit we use [mm]
    axis_normalization = np.asarray(
        [
            metadata["qr_data"]["pix_size"] * 1000,
            metadata["qr_data"]["pix_size"] * 1000,
            1.0 / metadata["fps"],
            1.0 / metadata["fps"],
        ]
    )

    X_px_fr = _get_X_px_fr_more_tools(data, incision_bboxes, tool_indexes, time_axis=time_axis)
    X_px_fr = add_dim_with_cumulative_number_of_empty_frames(X_px_fr, empty_frames_axis=empty_frame_axis)
    X = X_px_fr * axis_normalization

    if n_clusters > 1:
        ms = KMeans(n_clusters=n_clusters)
        ms.fit(X)
        labels = ms.labels_
        # labels = ms.predict(X)
        cluster_centers = ms.cluster_centers_

        #n_neighbors should be half of the less frequent label
        n_per_class = np.bincount(labels)
        logger.debug(f"{np.bincount(labels)=}")
        if np.min(n_per_class) > 2:
            n_neighbors = int(np.min(n_per_class) / 2)
            if n_neighbors > 100:
                n_neighbors = 100

            labels = _smooth_in_1D(X, labels, time_axis=time_axis, n_neighbors=n_neighbors)
        splits_s, splits_frames = _get_splits(X, labels, metadata["fps"], time_axis=time_axis, weight_of_later=weight_of_later)
    else:
        # splits_s = [np.mean(X[:, time_axis])]
        # splits_frames = [int(splits_s[0] * float(metadata["fps"]))]
        splits_s = []
        splits_frames = []
        cluster_centers = [np.mean(X, axis=0)]
        labels = np.zeros(X.shape[0])


    # print(f"{splits_s=}")
    # print(f"{splits_frames=}")

    X_px_fr = _get_X_px_fr_more_tools(data, oa_bbox, trim_tool_indexes, time_axis=time_axis)
    X_px_fr = add_dim_with_cumulative_number_of_empty_frames(X_px_fr, empty_frames_axis=empty_frame_axis)
    X2 = X_px_fr * axis_normalization

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


    if plot_clusters:
        plot_track_clusters(
            X,
            labels,
            cluster_centers,
            new_splits_s,
            splits_s,
            X2,
            clusters_image_path=clusters_image_path,
        )


    return new_splits_s, new_splits_frames





def plot_track_clusters(
    X, labels,
    cluster_centers, splits_s,
    splits_hints_s, X2,
    clusters_image_path: Optional[Path] = None
):
    from matplotlib import pyplot as plt

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    # fig = plt.figure(figsize=(15,4))

    fig, ((a0, a1), (a2, a3)) = plt.subplots(2, 2,
                                     gridspec_kw={'width_ratios': [1, 3]}
                                    )
    fig.delaxes(a2)
    # print("fig created")
    # plt.subplot(211)
    # plt.subplot(121)
    # plt.clf()

# toodo
    _draw_points(a0, cluster_centers, X, X2, n_clusters_, labels, ax0=1, ax1=0)
    a0.set_xlabel("[mm]")
    a0.set_ylabel("[mm]")
    # plt.title("Estimated number of clusters: %d" % n_clusters_)

    # plt.subplot(212)
    # plt.subplot(122)
    # colors = [
    #     "#dede00",
    #     "#377eb8",
    #     "#f781bf",
    #     "#81bf37",
    #     "#bf3781",
    #     "#f3781b",
    #     "#eb88b1",
    #     "#1bff78",
    # ]

#     ax0 = 2
#     ax1 = 0
    _draw_points(a1, cluster_centers, X, X2, n_clusters_, labels, ax0=2, ax1=0)


    # # draw additional points used for cropping
    # a1.plot(X2[:,ax0],X2[:,ax1], ".", color="k")
    # for k, col in zip(range(n_clusters_), colors):
    #     my_members = labels == k
    #     cluster_center = cluster_centers[k]
    #     a1.plot(X[my_members, ax0], X[my_members, ax1], markers_x[k], color=col)
    #     a1.plot(
    #         cluster_center[ax0],
    #         cluster_center[ax1],
    #         markers[k],
    #         markerfacecolor=col,
    #         markeredgecolor="k",
    #         markersize=14,
    #     )
    a1.set_xlabel("[s]")
    _draw_time_lines_in_plot(a1, splits_s, splits_hints_s)

    _draw_points(a3, cluster_centers, X, X2, n_clusters_, labels, ax0=2, ax1=3)
    _draw_time_lines_in_plot(a3, splits_s, splits_hints_s)
    a3.set_xlabel("[s]")
    a3.set_ylabel("[s]")

#     a2.plot(X2[:,ax0],X2[:,ax1], ".", color="k")
#     for k, col in zip(range(n_clusters_), colors):
#         my_members = labels == k
#         cluster_center = cluster_centers[k]
#         a2.plot(X[my_members, ax0], X[my_members, ax1], markers_x[k], color=col)
#         a2.plot(
#             cluster_center[ax0],
#             cluster_center[ax1],
#             markers[k],
#             markerfacecolor=col,
#             markeredgecolor="k",
#             markersize=14,
#         )

#     ax0 = 2
#     ax1 = 3
#     a2.set_xlabel("[s]")
#     colors = ["g", 'r']
#     linestyles= [(0, (4, 8)), (6, (4, 8))]
#     for i, yline in enumerate(splits_s):
#         # if odd use green, if even use red
#         a1.axvline(x=yline, c=colors[i%2], linestyle=linestyles[i%2], linewidth=1)
#         # plt.axhline(y=yline, c=colors[i%2])
#     for i, yline in enumerate(splits_hints_s):
#         a2.axvline(x=yline, c="k", linestyle=":", linewidth=1)


    if clusters_image_path is not None:
        plt.savefig(clusters_image_path)
        plt.close(fig)

def _draw_points(a0, cluster_centers, X, X2, n_clusters_, labels, ax0=1, ax1=0):
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
    markers_x = ["." for i in range(len(colors))]
    markers_o = ["." for i in range(len(colors))]
    logger.debug(f"{cluster_centers=}")

    # draw additional points used for cropping
    a0.plot(X2[:,ax0],X2[:,ax1], ".", color="k")

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        a0.plot(X[my_members, ax0], X[my_members, ax1], markers_x[k], color=col)
        a0.plot(
            cluster_center[ax0],
            cluster_center[ax1],
            markers[k],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )


def _draw_time_lines_in_plot(a1, splits_s, splits_hints_s):
    colors = ["g", 'r']
    linestyles= [(0, (4, 8)), (6, (4, 8))]
    for i, yline in enumerate(splits_s):
        # if odd use green, if even use red
        a1.axvline(x=yline, c=colors[i%2], linestyle=linestyles[i%2], linewidth=1)
        # plt.axhline(y=yline, c=colors[i%2])
    for i, yline in enumerate(splits_hints_s):
        a1.axvline(x=yline, c="k", linestyle=":", linewidth=1)


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
