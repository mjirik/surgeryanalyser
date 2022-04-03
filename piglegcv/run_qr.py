import sys
import os
import cv2
import json
import numpy as np
from pyzbar.pyzbar import decode
from loguru import logger


def save_json(data: dict, output_json: str):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as output_file:
        json.dump(data, output_file)


def main_qr(filename, output_dir):
    
    cap = cv2.VideoCapture(str(filename))
    QRinit = False
    pix_size = 1.0
    qr_size = 0.027
    is_detected = 0
    img_first = None
    box = []
    qr_text = None
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        #try read QR code
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = decode(grey)
        logger.debug(res)
        # if len(res) > 0:
        for oneqr in res:
            txt = oneqr.data.decode("utf8")
            logger.debug(f"qr code text = {txt}")
            if txt == "Resolution 30 mm":
                qr_size = 0.030
            elif txt == "QR scale pigleg":
                qr_size = 0.027
            else:
                continue
            is_detected = 1
            a = np.array(oneqr.polygon[0])
            b = np.array(oneqr.polygon[1])
            #print(a,b)
            pix_size = qr_size / np.linalg.norm(a-b)
            #print(pix_size)
            img_first = img
            box = [[point.x, point.y] for point in oneqr.polygon]
            break
            
    qr_data = {}
    qr_data['is_detected'] = is_detected
    qr_data['box'] = box
    qr_data['pix_size'] = pix_size
    qr_data['qr_size'] = qr_size
    qr_data['text'] = qr_text

    # save QR to the json file
    save_json({"qr_data": qr_data}, os.path.join(output_dir, "qr_data.json"))

   
if __name__ == '__main__':

    main_qr(sys.argv[1], sys.argv[2])
