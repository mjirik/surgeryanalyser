from pathlib import Path
import cv2
import json
import loguru
from loguru import logger
from typing import Optional
import shutil
import traceback
import time
import subprocess
#try:
#    from .run_tracker_lite import main_tracker
#    from .run_tracker_bytetrack import main_tracker_bytetrack
#    from .run_mmpose import main_mmpose
#    from .run_qr import main_qr
#    from .run_report import main_report
#    from .run_perpendicular import main_perpendicular, get_frame_to_process
#    from .incision_detection_mmdet import run_incision_detection
#except ImportError as e:
#    logger.debug(e)
#    from run_tracker_lite import main_tracker
from run_tracker_bytetrack import main_tracker_bytetrack
#from run_mmpose import main_mmpose
from run_qr import main_qr
import run_qr
from run_report import main_report
from run_perpendicular import main_perpendicular, get_frame_to_process
from tools import save_json
import numpy as np
from incision_detection_mmdet import run_incision_detection
# from run_qr import bbox_info_extraction_from_frame


class DoComputerVision():
    def __init__(self, filename: Path, outputdir: Path, meta: Optional[dict] = None):
        self.filename:Path = Path(filename)
        self.outputdir:Path = Path(outputdir)
        self.meta:dict = meta
        self.logger_id = None
        self.frame:Optional[np.ndarray] = None
        self.filename_cropped:Optional[Path] = None

        log_format = loguru._defaults.LOGURU_FORMAT
        self.logger_id = logger.add(
            str(Path(outputdir) / "piglegcv_log.txt"),
            format=log_format,
            level="DEBUG",
            rotation="1 week",
            backtrace=True,
            diagnose=True,
        )


    def run(self):
        self.meta = {}
        logger.debug(f"CV processing started on {self.filename}, outputdir={self.outputdir}")

        try:
            if Path(self.filename).suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
                self.run_image_processing()
            else:
                #run_video_processing(filename, outputdir)
                self.run_video_processing()
            save_json(self.meta, Path(self.outputdir) / "meta.json")

            logger.debug("Work finished")
        except Exception as e:
            logger.error(traceback.format_exc())
        logger.remove(self.logger_id)

    def run_image_processing(self):
        logger.debug("Running image processing...")
        self.frame = get_frame_to_process(str(self.filename_cropped))
        qr_data = run_qr.bbox_info_extraction_from_frame(self.frame)
        qr_data['qr_scissors_frames'] = []
        self.meta["qr_data"] = qr_data

        main_perpendicular(self.filename, self.outputdir, self.meta)
        logger.debug("Perpendicular finished.")

    def run_video_processing(self):

        """

        :param filename:
        :param outputdir:
        :param meta: might be used for progressbar
        :return:
        """
        logger.debug("Running video processing...")
        if self.meta is None:
            meta = {}

        # get_sigle_frame
        # single_frame_processing ->
        # video_preprocessing - rotate, rescale and crop -> file
        # single_frame_processing on rotated
        # bytrack
        # make_report

        s = time.time()
        self.frame = get_frame_to_process(str(self.filename))
        qr_data = run_qr.bbox_info_extraction_from_frame(self.frame)
        qr_data['qr_scissors_frames'] = []
        logger.debug(f"Single frame processing on original mediafile finished in {time.time() - s}s.")

        # video_preprocessing - rotate, rescale and crop -> file
        s = time.time()
        self.filename_cropped = self.rotate_rescale_crop()
        logger.debug(f"Cropping done in {time.time() - s}s.")

        s = time.time()
        self.run_image_processing()
        logger.debug(f"Single frame processing on cropped mediafile finished in {time.time() - s}s.")

        logger.debug(f"Image processing finished in {time.time() - s}s.")

        s = time.time()
        main_tracker_bytetrack(
            config_file="./resources/tracker_model_bytetrack/bytetrack_pigleg.py",
            filename=self.filename,
            output_dir=self.outputdir,
            checkpoint=Path(__file__).parent / "resources/tracker_model_bytetrack/epoch.pth",
            device="cuda"
        )
        logger.debug(f"Tracker finished in {time.time() - s}s.")

        logger.debug(f"filename={self.filename}, outputdir={self.outputdir}")
        logger.debug(f"filename={Path(self.filename).exists()}, outputdir={Path(self.outputdir).exists()}")

        s = time.time()
        main_report(self.filename, self.outputdir, self.meta)
        logger.debug(f"Report finished in {time.time() - s}s.")

        logger.debug("Report based on video is finished.")
        logger.debug("Video processing finished")


    def rotate_rescale_crop(self, transpose=False):
        # base_name, extension = str(self.filename).rsplit('.', 1)

        self.filename_cropped = self.outputdir / "cropped.mp4"

        # Recreate the modified file path
        # new_file_path = new_base_name + '.' + "mp4"
        logger.debug(f"self.filename_cropped={self.filename_cropped}")

        # s = ["ffmpeg", '-i', str(self.filename), '-ac', '2', "-y", "-b:v", "2000k", "-c:a", "aac", "-c:v", "libx264", "-b:a", "160k",
        #      "-vprofile", "high", "-bf", "0", "-strict", "experimental", "-f", "mp4", base_name]

        filter_str = ''
        cr_out_w = 400
        cr_out_h = 100
        cr_x = 0
        cr_y = 0
        if False:
            filter_str += f"crop={cr_out_w}:{cr_out_h}:{cr_x}:{cr_y},"
        if transpose:
            filter_str += "transpose=1,"

        filter_str += 'scale="720:trunc(ow/a/2)*2"'

        logger.debug(f"filename={self.filename}, {self.filename.exists()}")
        s = ["ffmpeg", '-i', str(self.filename),
             '-filter:v', filter_str, "-an", "-y", "-b:v", "500k",
             str(self.filename_cropped)
             ]
        logger.debug(f"{' '.join(s)}")
        # p = subprocess.Popen(s)
        # p.wait()
        subprocess.check_output(s)

        logger.debug(f"filename_cropped={self.filename_cropped}, {self.filename_cropped.exists()}")
        return self.filename_cropped
        # return self.filename


def do_computer_vision(filename, outputdir, meta):
    DoComputerVision(filename, outputdir, meta).run()


def do_computer_vision_2(filename, outputdir, meta):
    log_format = loguru._defaults.LOGURU_FORMAT
    logger_id = logger.add(
        str(Path(outputdir) / "piglegcv_log.txt"),
        format=log_format,
        level="DEBUG",
        rotation="1 week",
        backtrace=True,
        diagnose=True,
    )
    logger.debug(f"CV processing started on {filename}, outputdir={outputdir}")

    try:
        if Path(filename).suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
            run_image_processing(filename, outputdir)
        else:
            #run_video_processing(filename, outputdir)
            run_video_processing2(filename, outputdir)

        logger.debug("Work finished")
    except Exception as e:
        logger.error(traceback.format_exc())
    logger.remove(logger_id)


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

def run_video_processing2(filename: Path, outputdir: Path, meta:dict=None) -> dict:
    """

    :param filename:
    :param outputdir:
    :param meta: might be used for progressbar
    :return:
    """
    logger.debug("Running video processing...")
    if meta is None:
        meta = {}
    s = time.time()
    main_qr(filename, outputdir)
    logger.debug(f"QR finished in {time.time() - s}s.")
    run_image_processing(filename, outputdir, skip_qr=True)
    s = time.time()
    logger.debug(f"Image processing finished in {time.time() - s}s.")

    # main_tracker_bytetrack("\"{}\" \"{}\" \"{}\" --output_dir \"{}\"".format('./resources/tracker_model_bytetrack/bytetrack_pigleg.py','./resources/tracker_model_bytetrack/epoch.pth', filename, outputdir))
    # f"\"./resources/tracker_model_bytetrack/bytetrack_pigleg.py\" \"{filename}\" --output_dir \"{outputdir}\"",
    main_tracker_bytetrack(
        config_file="./resources/tracker_model_bytetrack/bytetrack_pigleg.py",
        filename=filename,
        output_dir=outputdir,
        checkpoint=Path(__file__).parent / "resources/tracker_model_bytetrack/epoch.pth",
        device="cuda"
    )
    # run_media_processing(Path(filename), Path(outputdir))
    logger.debug(f"Tracker finished in {time.time() - s}s.")

    #
    # s = time.time()
    # main_mmpose(filename, outputdir)
    # logger.debug(f"MMpose finished in {time.time() - s}s.")

    logger.debug(f"filename={filename}, outputdir={outputdir}")
    logger.debug(f"filename={Path(filename).exists()}, outputdir={Path(outputdir).exists()}")

    main_report(filename, outputdir)
    
    logger.debug("Report based on video is finished.")

    # if extention in images_types:

    # logger.debug("Perpendicular finished.")
    logger.debug("Video processing finished")


def run_image_processing(filename: Path, outputdir: Path, skip_qr=False) -> dict:
    logger.debug("Running image processing...")
    frame = get_frame_to_process(str(filename))
    run_qr.bbox_info_extraction_from_frame(frame)
    main_perpendicular(filename, outputdir)
    logger.debug("Perpendicular finished.")
    # TODO add predict image
    # img = mmcv.imread(str(img_fn))
    # run_incision_detection(filename, outputdir)


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

    metadata = {"filename_full": str(filename), "fps": fps, "frame_count": totalframecount}
    json_file = outputdir / "meta.json"
    save_json(metadata, json_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process media file")
    parser.add_argument("filename", type=str)
    parser.add_argument("outputdir", type=str)
    args = parser.parse_args()
    do_computer_vision(Path(args.filename), Path(args.outputdir))
