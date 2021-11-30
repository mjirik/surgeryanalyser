import os
import cv2
import json
import torch
import argparse
import numpy as np
import shlex
from pyzbar.pyzbar import decode

#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from detectron2.data import transforms as T

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from extern.sort import sort


CFG = {
    "output_dir": "./__OUTPUT__/",
    "SORT": {"max_age": 4, "min_hits": 6, "iou_threshold": 0.0},  # int  # int  # float
}


class CustomPredictor(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR", "L"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]

            image = (
                T.Resize(
                    (
                        int(
                            height
                            * CFG["R-CNN"]["INPUT"]["RESIZE"]["RELATIVE"]["ratio"]
                        ),
                        int(
                            width * CFG["R-CNN"]["INPUT"]["RESIZE"]["RELATIVE"]["ratio"]
                        ),
                    )
                )
                .get_transform(original_image)
                .apply_image(original_image)
            )

            if self.input_format == "L":
                image = torch.as_tensor(np.ascontiguousarray(image))
            else:
                image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            return predictions


def merge_config():
    if args.config is not None:
        with open(args.config, "r") as json_file:
            CFG.update(json.load(json_file))


def get_detectron_cfg() -> CfgNode:
    """
    Merge the local config file with the detectron2 one.

    :return: detectron2 config node
    """

    # initialize the configuration
    cfg = get_cfg()

    # model specification
    cfg.merge_from_file(
        model_zoo.get_config_file(f"COCO-Detection/{CFG['R-CNN']['model']}.yaml")
    )
    cfg.MODEL.WEIGHTS = CFG["R-CNN"]["weights"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = CFG["DATA"]["num_classes"]

    # proposals of bounding boxes
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = CFG["R-CNN"]["ANCHOR"]["aspect_ratios"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = CFG["R-CNN"]["ANCHOR"]["sizes"]

    # input image
    cfg.INPUT.FORMAT = CFG["R-CNN"]["INPUT"]["format"]

    # which dataset shall be used for training and validation
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()

    # evaluation
    cfg.TEST.DETECTIONS_PER_IMAGE = CFG["DATA"]["max_dets"]

    # dataloader
    cfg.DATALOADER.NUM_WORKERS = CFG["DATA"]["num_workers"]

    # batch size
    cfg.SOLVER.IMS_PER_BATCH = CFG["SOLVER"]["batch_size"]

    ## output path
    cfg.OUTPUT_DIR = CFG["output_dir"]
    #prefix = "" if (CFG["OUTPUT"]["prefix"] == "") else f"{CFG['OUTPUT']['prefix']}_"
    #suffix = "" if (CFG["OUTPUT"]["suffix"] == "") else f"_{CFG['OUTPUT']['suffix']}"
    #cfg.OUTPUT_DIR = os.path.join(
        #CFG["output_dir"], "tracker", f"{prefix}{CFG['R-CNN']['model']}{suffix}"
    #)

    ## check if output path exists
    #output_orig = cfg.OUTPUT_DIR
    #idx = 1
    #while os.path.exists(cfg.OUTPUT_DIR):
        #cfg.OUTPUT_DIR = f"{output_orig}__{idx}"
        #idx += 1

    return cfg


def read_img(img_name: str):
    if CFG["R-CNN"]["INPUT"]["format"] == "L":  # greyscale
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_name)

    return img


def save_json(data: dict, output_json: str):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as output_file:
        json.dump(data, output_file)

#pix_size in m
def create_pdf_report(track, image, source_fps, pix_size, output_file_name, output_file_name2, ds_threshold = 0.1):

    N = len(track)
    data_pixel = []
    frame_id = []
    for i, frame in enumerate(track):
        if frame != []:
            #print(frame)
            #exit()
            box = np.array(frame[0])
            #print(i)
            frame_id.append(i)
            position = np.array([np.mean([box[0],box[2]]), np.mean([box[1],box[3]])])
            data_pixel.append(position)



    data_pixel = np.array(data_pixel)
    data = pix_size * data_pixel
    t = 1.0/source_fps * np.array(frame_id)
    dxy = data[1:] - data[:-1]
    ds = np.sqrt(np.sum(dxy*dxy, axis=1))

    ds[ds>ds_threshold] = 0
    dt = t[1:] - t[:-1]
    #print(dt)
    L = np.sum(ds)
    T = np.sum(dt)

    fig = plt.figure()
    fig.suptitle('Space trajectory analysis of needle holder', fontsize=14, fontweight='bold')
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)
    ax.set_title('Plot on the scene image')

    ax.imshow(image[:,:,::-1])
    box_text = 'Total in-plain track {:.2f} m / {:.2f} sec'.format(L, T)
    ax.text(100, 150, box_text, style='italic', bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 10})

    ax.plot(data_pixel[:,0], data_pixel[:,1],'+b', markersize=12)
    x = data_pixel[0, 0]
    y = data_pixel[0, 1]
    ax.plot(x, y,'go')
    ax.annotate('Start', xy=(x, y), xytext=(x+100, y-100), arrowprops=dict(facecolor='black', shrink=0.001))
    x = data_pixel[-1, 0]
    y = data_pixel[-1, 1]
    ax.plot(x, y,'ro')
    ax.annotate('Stop', xy=(x, y), xytext=(x+100, y+100), arrowprops=dict(facecolor='black', shrink=0.001))
    ax.axis('off')
    #ax.plot(x[-1], y[-1],'ro')
    #plt.plot(t, dist,'-')

    #plt.show()
    plt.savefig(output_file_name)

    fig = plt.figure()
    fig.suptitle('Time analysis', fontsize=14, fontweight='bold')
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)
    ax.set_title('Actual in-plain position of needle holder')
    ax.set_xlabel('Time [sec]')
    #ax.set_ylabel('Data')
    #ax.plot(t, data[:, 1], "-+r", label="X coordinate [mm]"  )
    #ax.plot(t, data[:, 0], "-+b", label="Y coordinate [m]"  )
    ax.plot(t[0:-1], np.cumsum(ds), "-k", label="Track [m]"  )
    ax.plot(t[0:-1],gaussian_filter(ds/dt, sigma=2) , ":g", label="Velocity [m/sec]"  )
    ax.legend(loc="upper left")
    #plt.plot(t_gt, y_gt, 'b')
    #plt.plot(t, x, 'r:')
    #plt.plot(t, y, 'b:')

    #plt.show()
    plt.savefig(output_file_name2)


def tracking_sort(
    predictor: CustomPredictor, tracker: sort.Sort, filename: str, output_dir: str
):
    track_id_last = 1
    final_tracks = list()

    cap = cv2.VideoCapture(str(filename))
    source_fps = int(cap.get(cv2.CAP_PROP_FPS))

    QRinit = False
    DbgVidInit = False
    QRdet = cv2.QRCodeDetector()
    pix_size = 1.0
    j = 0
    while cap.isOpened():
        ret, img = cap.read()
        j += 1
        if not ret:
            break

        #if j > 150:
            #break

        # init
        if not QRinit:
            #try read QR code
            
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            res = decode(grey)
            if len(res) > 0:
                a = np.array(res[0].polygon[0])
                b = np.array(res[0].polygon[1])
                #print(a,b)
                pix_size = 0.027 / np.linalg.norm(a-b)
                #print(pix_size)
                QRinit = False
                img_first = img
            
            ###############################
            #OpenCV encodes the frames in the BGR order by default.
            #retval, points, _ = QRdet.detectAndDecode(img[:,:,::-1])
            #print(retval, points)
            #return()
            #if points is not None:
                #pix_size = 27.0 / np.linalg.norm(points[0,0:1,:]-points[0,1:2,:])
                #print('pix_size', pix_size)
                #QRinit = False
                #img_first = img
        
        if not DbgVidInit:
            img_first = img
            # codec selection
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            height, width, channels = img.shape
            video = cv2.VideoWriter(
                os.path.join(output_dir, "video.avi"), fourcc, source_fps, (width, height)
            )
            DbgVidInit = True

        #predict
        outputs = predictor(img)
        outputs = outputs["instances"].to("cpu")

        # filter unwanted detections (only when any detection exists)
        dets = outputs.pred_boxes.tensor.numpy()

        # filter the detections
        if len(dets) > 0:
           scores = outputs.scores.numpy()
           det = []
           for s, d in zip(scores, dets):
              if (max(scores) >= 0.95) and (s < 0.95):
                 break
              elif (max(scores) >= 0.80) and (s < 0.80):
                 break
              elif (max(scores) >= 0.50) and (s < 0.50):
                 break
              elif (max(scores) >= 0.20) and (s < 0.20):
                 break
              det.append(d)
           dets = np.array(det)

        # update SORT
        tracks = tracker.update(dets)

        # check if the last track id is in the list of all tracks for current image
        skip_wrong_tracks = True if track_id_last in [i[4] for i in tracks] else False

        # actualize the track id when it was lost with Kalman Filter
        if (len(tracks) == 1) and skip_wrong_tracks is False:
            track_id_last = tracks[0][4]

        # tracks are in the format: [x_tl, y_tl, x_br, y_br, track_id], where tl is top-left and br is bottom-right
        filtered_tracks = list()
        for track in tracks:
            if (skip_wrong_tracks is True) and (track[4] != track_id_last):
                continue

            # save the tracks after filtering
            filtered_tracks.append(track.tolist())

            # color
            color = (0, 255, 0)

            # draw detection
            cv2.rectangle(
                img,
                (int(track[0]) - 1, int(track[1]) - 1),
                (int(track[2]) - 1, int(track[3]) - 1),
                color,
                thickness=2,
            )

            # draw track ID, coordinates: bottom-left
            cv2.putText(
                img,
                str(track[4]),
                (int(track[0]) - 2, int(track[3]) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=color,
                thickness=2,
            )

        # save image to the video
        video.write(img)
        
        # store the final tracks to the list
        final_tracks.append(filtered_tracks)

    # destroy all windows and release the video
    # cv2.destroyAllWindows()
    if DbgVidInit:
        video.release()

    # save the final tracks to the json file
    save_json({"tracks": final_tracks}, os.path.join(output_dir, "tracks.json"))

    #process data
    create_pdf_report(final_tracks, img_first, source_fps, pix_size, os.path.join(output_dir, "report_1.jpg"), os.path.join(output_dir, "report_2.jpg"))



#if __name__ == "__main__":
def main_tracker(commandline):
    print("main_tracker: initiated")
    # Parse commandline
    parser = argparse.ArgumentParser(
        description="Tracking the objects using MOT methods."
    )

    # Optional arguments
    parser.add_argument(
        "-pre", "--prefix", type=str, default="", help="Prefix for the output."
    )
    parser.add_argument(
        "-suf", "--suffix", type=str, default="", help="Suffix for the output."
    )
    parser.add_argument(
        "-out", "--output_dir", type=str, default="", help="Output directory."
    )

    # Positional arguments
    parser.add_argument(
        "model_dir", type=str, help="Path to the directory with a model."
    )
    parser.add_argument("filename", type=str, help="video file.")

    # Parsing arguments
    args = parser.parse_args(shlex.split(commandline))

    # ==================================================================================================================

    # update a config
    with open(os.path.join(args.model_dir, "config.json"), "r") as json_file:
        CFG.update(json.load(json_file))

    if os.path.exists(os.path.join(args.model_dir, "last_checkpoint")):
        with open(
            os.path.join(args.model_dir, "last_checkpoint"), "r"
        ) as checkpoint_file:
            CFG["R-CNN"]["weights"] = os.path.join(
                args.model_dir, checkpoint_file.readline()
            )

    # update the optional arguments
    if args.prefix is not "":
        CFG["OUTPUT"]["prefix"] = args.prefix

    if args.suffix is not "":
        CFG["OUTPUT"]["suffix"] = args.suffix

    if args.output_dir is not "":
        CFG["output_dir"] = args.output_dir

    # get the detectron2 configuration and create an output directory
    cfg = get_detectron_cfg()

    
    #os.makedirs(cfg.OUTPUT_DIR)

    # save the used configuration (for training, testing...)
    save_json(CFG, os.path.join(cfg.OUTPUT_DIR, "config.json"))

    # initialize the predictor
    predictor = CustomPredictor(cfg)

    # use the SORT method for tracking the objects
    mot_tracker = sort.Sort(
        max_age=CFG["SORT"]["max_age"],
        min_hits=CFG["SORT"]["min_hits"],
        iou_threshold=CFG["SORT"]["iou_threshold"],
    )

    # get tracks
    tracking_sort(predictor, mot_tracker, args.filename, cfg.OUTPUT_DIR)
