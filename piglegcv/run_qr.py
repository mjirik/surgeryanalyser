import sys
import os
import cv2
import json
import numpy as np
from loguru import logger
from tools import save_json, load_json
from pathlib import Path
from qreader import QReader

def main_qr(filename, output_dir):
    logger.debug("looking for qr code...")
    qreader = QReader()
    
    cap = cv2.VideoCapture(str(filename))
    pix_size = 1.0
    qr_size = 0.027
    is_detected = False
    box = []
    qr_text = None
    qr_scissors_frames= []
    i = -1
    first_img = False
    image_processing_step = 10
    while cap.isOpened():
        i += 1
        ret = cap.grab()
        
        if not ret:
            break
        
        if not i % image_processing_step:
            _, img = cap.retrieve()
        
            if not first_img:
                first_img = True
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

            #try read QR code
            detected_qr_codes = qreader.detect_and_decode(image=img, return_bboxes=True)

            for bbox, _oneqr in detected_qr_codes:
                if _oneqr != None:
                    oneqr, resize_factor = _oneqr
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
                    if not is_detected:
                        is_detected = True

                        # qreader returns detection bbox to input image, pyzbar reactangle and polygon to the bbox crop resized by resize_factor -> 
                        # qr code polygon in input image = polygon / resize_facor + bbox 
                        box = [[int(point.x / resize_factor + bbox[0]), int(point.y / resize_factor + bbox[1])] for point in oneqr.polygon]

                        # debug only
                        #cv2.drawContours(img, [np.asarray(box)], 0, (0, 255, 0), 2)
                        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        #cv2.imshow("image", img)
                        #cv2.waitKey(0)
                        #cv2.imwrite(filename='_image.jpeg', img=img)
                        # debug only

                        a = np.array(box[0])
                        b = np.array(box[1])
                        pix_size = qr_size / np.linalg.norm(a-b)
                
            
    qr_data = {}

# todo use the pigleg holder detection based estimator
    qr_data["pix_size_method"] = "QR" if is_detected else "video size estimation"
    if True:
        # pigleg_holder_width [m] - usually it takes around half of the image width
        scene_size = 0.300 # [m]
        size_by_scene = scene_size / width

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
