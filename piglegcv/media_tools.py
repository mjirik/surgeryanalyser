import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger


# Function to handle serialization
def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (set, complex)):
        return list(obj)  # Convert sets and complex numbers to lists
    elif isinstance(obj, bytes):
        return obj.decode()  # Convert bytes to string
    else:
        return str(obj)  # Convert other non-serializable types to string


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(data: dict, output_json: Union[str, Path], update: bool = True):
    logger.debug(f"Writing '{output_json}'")

    output_json = Path(output_json)
    output_json.parent.mkdir(exist_ok=True, parents=True)
    # os.makedirs(os.path.dirname(output_json), exist_ok=True)
    dct = {}
    if update and output_json.exists():
        with open(output_json, "r") as output_file:
            dct = json.load(output_file)
        logger.debug(f"old keys: {list(dct.keys())}")
    dct.update(data)
    logger.debug(f"updated keys: {list(dct.keys())}")
    with open(output_json, "w") as output_file:
        try:
            json.dump(dct, output_file, indent=4,
                      # cls=NumpyEncoder,  # here is necessary to solve all types of objects
                      default=serialize  # here we are solving only the non serializable objects
                      )

        except Exception as e:
            logger.error(f"Error writing json file {output_json}: {e}")
            logger.error(f"Data: {dct}")
            print_nested_dict_with_types(dct, 4)

            raise e


def load_json(filename: Union[str, Path]):
    filename = Path(filename)
    if os.path.isfile(filename):
        with open(filename, "r") as fr:
            try:
                data = json.load(fr)
            except ValueError as e:
                return {}
            return data
    else:
        return {}

def print_nested_dict_with_types(d: dict, indent: int = 0):
    for k, v in d.items():
        if isinstance(v, dict):
            logger.debug(f"{' ' * indent}{k}:")
            print_nested_dict_with_types(v, indent + 2)
        else:
            logger.debug(f"{' ' * indent}{k}: {type(v)}")


def make_images_from_video(
    filename: Path,
    outputdir: Path,
    n_frames=None,
    scale=1,
    filemask: str = "{outputdir}/frame_{frame_id:0>6}.png",
    width: Optional[int] = None,
    height: Optional[int] = None,
    make_square: bool = False,
    create_meta_json: bool = True,
) -> Path:
    import cv2

    outputdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(filename))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width:
        scale = None
    if height:
        scale = None

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            logger.warning(f"Reading frame {frame_id} in {str(filename)} failed.")
            break
        if scale is None and width is not None:
            scale = width / frame.shape[1]
        if scale is None and height is not None:
            scale = height / frame.shape[0]

        frame_id += 1
        if frame_id > n_frames:
            break
        if not ret:
            break
        else:
            file_name = filemask.format(outputdir=outputdir, frame_id=frame_id)
            frame = rescale(frame, scale)
            if make_square:
                frame = crop_square(frame)
            cv2.imwrite(file_name, frame)
            logger.trace(file_name)
    cap.release()

    if create_meta_json:
        metadata = {
            "filename_full": str(filename),
            "fps": fps,
            "frame_count": totalframecount,
        }
        json_file = outputdir / "meta.json"
        save_json(metadata, json_file)


def rescale(frame, scale):
    import cv2

    if scale != 1:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return frame


def convert_avi_to_mp4(avi_file_path, output_name):
    s = [
        "ffmpeg",
        "-i",
        avi_file_path,
        "-ac",
        "2",
        "-y",
        "-b:v",
        "2000k",
        "-c:a",
        "aac",
        "-c:v",
        "libx264",
        "-b:a",
        "160k",
        "-vprofile",
        "high",
        "-bf",
        "0",
        "-strict",
        "experimental",
        "-f",
        "mp4",
        output_name,
    ]
    subprocess.call(s)
    return True


def crop_square(frame: np.ndarray) -> np.ndarray:

    mn = np.min(frame.shape[:2])
    sh0 = frame.shape[0]
    sh1 = frame.shape[1]
    if sh0 > sh1:
        st0 = int((sh0 / 2) - (sh1 / 2))
        st1 = 0
    else:
        st0 = 0
        st1 = int((sh1 / 2) - (sh0 / 2))

    frame = frame[st0 : st0 + mn, st1 : st1 + mn]

    return frame
