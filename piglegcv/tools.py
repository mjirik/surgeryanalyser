from pathlib import Path
from typing import Union
import json
import os
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import cv2


 

def save_json(data:dict, output_json:Union[str,Path], update:bool=True):
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
        json.dump(dct, output_file, indent=4)


def load_json(filename:Union[str,Path]):
    filename = Path(filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as fr:
            try:
                data = json.load(fr)
            except ValueError as e:
                return {}
            return data
    else:
        return {}


def unit_conversion(value, input_unit:str, output_unit:str):
    in_kvantif = input_unit[-2] if len(input_unit) > 1 else ""
    out_kvantif = output_unit[-2] if len(output_unit) > 1 else ""

    in_k = _unit_multiplier(in_kvantif)
    out_k = _unit_multiplier(out_kvantif)

    return value * in_k / out_k


def _unit_multiplier(kvantif:str):
    multiplier = None
    if len(kvantif) == 0:
        multiplier = 1
    elif kvantif == "u":
        multiplier = 1e-6
    elif kvantif == "m":
        multiplier = 1e-3
    elif kvantif == "c":
        multiplier = 1e-2
    elif kvantif == "k":
        multiplier = 1e+3
    elif kvantif == "M":
        multiplier = 1e+6
    elif kvantif == "G":
        multiplier = 1e+9
    else:
        raise ValueError(f"Unknown unit kvantifier {kvantif}")

    return multiplier


def flatten_dict(dct:dict, parent_key:str='', sep:str='_') -> dict:
    """
    Flatten nested dictionary
    :param dct: nested dictionary
    :param parent_key: parent key
    :param sep: separator
    :return: flattened dictionary
    """
    items = []
    for k, v in dct.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def remove_empty_lists(dct:dict) -> dict:
    """
    Remove empty lists from dictionary
    :param dct: dictionary
    :return: dictionary without empty lists
    """
    return {k: v for k, v in dct.items() if v != []}

def draw_bbox_into_image(img:np.ndarray, bbox, linecolor=(255, 0, 0), linewidth=2, show_confidence=False) -> np.ndarray:
    if bbox is not None:
        bbox=np.asarray(bbox)
        x1, y1, x2, y2, confidence = bbox.astype(int).tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), linecolor, linewidth)
        if show_confidence:
            cv2.putText(img, f"{confidence:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, linecolor, 2)
    return img


def _find_largest_incision_bbox(bboxes):
    max_area = 0
    max_bbox = None
    for bbox in bboxes:
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        print(area)
        # area = bbox[2] * bbox[3]
        if area > max_area:
            max_area = area
            max_bbox = bbox
    return max_bbox


def count_points_in_bbox(points, bbox):
    """ Count points in bounding box.
    points: np.ndarray with shape [i,2]
    bbox: np.ndarray with values, [x1,y1,x2,y2, confidence]
    """
    count = 0
    for point in points:
        if point[0] >= bbox[0] and point[0] <= bbox[2] and point[1] >= bbox[1] and point[1] <= bbox[3]:
            count += 1
    return count

def filter_points_in_bbox(points, bbox):
    """ Filter points in bounding box.
    points: np.ndarray with shape [i,2]
    bbox: np.ndarray with values, [x1,y1,x2,y2, confidence]
    """
    # count = 0
    logger.debug(f"{bbox}")
    logger.debug(f"{points}")
    points_in_bbox = []
    for point in points:
        if point[0] >= bbox[0] and point[0] <= bbox[2] and point[1] >= bbox[1] and point[1] <= bbox[3]:
            points_in_bbox.append(point)
    return np.asarray(points_in_bbox)


def make_bbox_square_and_larger(bbox, multiplicator=1.):
    size = np.max(np.asarray([(bbox[3]) - (bbox[1]), (bbox[2]) - (bbox[0])]) * multiplicator)
    center = ((bbox[3] + bbox[1]) / 2., (bbox[2] + bbox[0]) / 2.)
    newbbox = [
        center[1] - (size / 2.), center[0] - (size / 2.),
        center[1] + (size / 2.), center[0] + (size / 2.),
        bbox[4]
    ]
    return newbbox

def sort_bboxes(bboxes):
    """Sort bboxes by their confidence score."""
    return bboxes[bboxes[:, 4].argsort()[::-1]]


def make_bbox_larger(bbox, multiplicator=2.):
    size = np.asarray([(bbox[3]) - (bbox[1]), (bbox[2]) - (bbox[0])]) * multiplicator
    center = ((bbox[3] + bbox[1]) / 2., (bbox[2] + bbox[0]) / 2.)
    newbbox = [
        center[1] - (size[1] / 2.), center[0] - (size[0] / 2.),
        center[1] + (size[1] / 2.), center[0] + (size[0] / 2.),
        bbox[4]
    ]
    return newbbox


def crop_image(img, bbox):
    imcr = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    return imcr


def draw_bboxes_plt(img, bboxes, color="r") -> plt.Figure:
    """
    Draw bounding boxes and their confidence scores on an image.
    
    Parameters:
    - image_path: Path to the image file.
    - bboxes: List of bounding boxes in the format [x_min, y_min, x_max, y_max, score].
    """
    
    # Load the image
#     img = Image.open(image_path)
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    for bbox in bboxes:
        x_min, y_min, x_max, y_max, score = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # Display the confidence score above the bounding box
        plt.text(x_min, y_min, f"{score:.2f}", bbox=dict(facecolor='red', alpha=0.5))

#     plt.show()

#     # Convert the figure to a numpy array
#     fig.canvas.draw()
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
#     plt.close(fig)  # Close the figure to free up memory
    return fig
    