import sys
import os
import cv2
import json
import numpy as np
from pyzbar.pyzbar import decode
from loguru import logger
from tools import save_json, load_json
from pathlib import Path



def main_qr(filename, output_dir):
    logger.debug("looking for qr code...")
    
    cap = cv2.VideoCapture(str(filename))
    QRinit = False
    pix_size = 1.0
    qr_size = 0.027
    is_detected = 0
    img_first = None
    box = []
    qr_text = None
    qr_scissors_frames= []
    i = -1
    while cap.isOpened():
        i += 1
        ret, img = cap.read()
        if not ret:
            break
        
        #try read QR code
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = decode(grey)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        # logger.debug(res)
        # if len(res) > 0:
        for oneqr in res:
            txt = oneqr.data.decode("utf8")
            logger.debug(f"qr code text = '{txt}', frame={i}")
            if txt == "Resolution 30 mm":
                qr_size = 0.030
                qr_text = txt
            elif txt == "QR scale pigleg":
                qr_size = 0.027
                qr_text = txt
            elif txt == "Scissors 30 mm":
                qr_scissors_frames.append(i)
                if qr_text is None:
                    # Use only if no Scale QR code was detected
                    qr_size = 0.030
                    qr_text = txt
            else:
                logger.debug(f"Unknown QR code with text='{txt}', on frame={i}")
                continue
            is_detected = 1
            a = np.array(oneqr.polygon[0])
            b = np.array(oneqr.polygon[1])
            #print(a,b)
            pix_size = qr_size / np.linalg.norm(a-b)
            #print(pix_size)
            img_first = img
            box = [[point.x, point.y] for point in oneqr.polygon]
            # break
            
    qr_data = {}

# todo use the pigleg holder detection based estimator
    qr_data["pix_size_method"] = "QR" if is_detected else "video size estimation"
    if True:

        # pigleg_holder_width [m] - usually it takes around half of the image width
        scene_size = 0.300 # [m]
        size_by_scene = scene_size / width
        # qr_text = "pixel size by scene"
        # box = []

    qr_data['is_detected'] = is_detected
    qr_data['box'] = box
    qr_data['pix_size'] = pix_size
    qr_data['qr_size'] = qr_size
    qr_data['size_by_scene'] = size_by_scene
    qr_data['text'] = qr_text
    qr_data['qr_scissors_frames'] = qr_scissors_frames

    # save QR to the json file
    json_file = Path(output_dir) / "meta.json"
    print(f"prepared to save to file {str(json_file)}")
    logger.debug(f"prepared to save to file {str(json_file)}")
    save_json({"qr_data": qr_data}, json_file)
    return qr_data

   
if __name__ == '__main__':

    main_qr(sys.argv[1], sys.argv[2])
