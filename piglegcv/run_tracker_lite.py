import os
import cv2
import json
import torch
import argparse
import numpy as np

#from matplotlib.backends.backend_pdf import PdfPages
#import matplotlib.pyplot as plt

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

    # output path
    prefix = "" if (CFG["OUTPUT"]["prefix"] == "") else f"{CFG['OUTPUT']['prefix']}_"
    suffix = "" if (CFG["OUTPUT"]["suffix"] == "") else f"_{CFG['OUTPUT']['suffix']}"
    cfg.OUTPUT_DIR = os.path.join(
        CFG["output_dir"], "tracker", f"{prefix}{CFG['R-CNN']['model']}{suffix}"
    )

    # check if output path exists
    output_orig = cfg.OUTPUT_DIR
    idx = 1
    while os.path.exists(cfg.OUTPUT_DIR):
        cfg.OUTPUT_DIR = f"{output_orig}__{idx}"
        idx += 1

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

def create_pdf_report(track, output_file_name):

    N = len(track)
    #data = []
    t = []
    x = []
    y = []
    for i, frame in enumerate(track):
        if frame != []:
            box = np.array(frame[0])
            #data.append([np.mean([box[0],box[2]]), np.mean([box[1],box[3]])])
            t.append(i)
            x.append(np.mean([box[0],box[2]]))
            y.append(np.mean([box[1],box[3]]))


    #plt.imshow(image[:,:,::-1])
    #plt.plot(x, y,'b')
    #plt.plot(x[0], y[0],'go')
    #plt.plot(x[-1], y[-1],'ro')
    #plt.show()

    #plt.plot(t_gt, x_gt, 'r')
    #plt.plot(t_gt, y_gt, 'b')
    #plt.plot(t, x, 'r:')
    #plt.plot(t, y, 'b:')

    #plt.show()


    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(output_file_name) as pdf:
        plt.figure(figsize=(3, 3))
        plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
        plt.title('Page One')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        plt.rc('text', usetex=True)
        plt.figure(figsize=(8, 6))
        x = np.arange(0, 5, 0.1)
        plt.plot(x, np.sin(x), 'b-')
        plt.title('Page Two')
        pdf.savefig()
        plt.close()

        plt.rc('text', usetex=False)
        fig = plt.figure(figsize=(4, 5))
        plt.plot(x, x*x, 'ko')
        plt.title('Page Three')
        pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
        plt.close()

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Multipage PDF Example'
        d['Author'] = u'Jouni K. Sepp\xe4nen'
        d['Subject'] = 'How to create a multipage pdf file and set its metadata'
        d['Keywords'] = 'PdfPages multipage keywords author title subject'
        d['CreationDate'] = datetime.datetime(2009, 11, 13)
        d['ModDate'] = datetime.datetime.today()


def tracking_sort(
    predictor: CustomPredictor, tracker: sort.Sort, filename: str, output_dir: str
):
    track_id_last = 1
    final_tracks = list()

    cap = cv2.VideoCapture(str(filename))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    init = True
    j = 0
    while cap.isOpened():
        ret, img = cap.read()
        j += 1
        if not ret:
            break

        if j > 150:
            break

        # init
        if init:
            init = False

            #QR code
            det = cv2.QRCodeDetector()
            #OpenCV encodes the frames in the BGR order by default.
            retval, points, _ = det.detectAndDecode(img[:,:,::-1])
            print(retval, points)
            #return()
            pix_size = 27.0 / np.linalg.norm(points[0,0:1,:]-points[0,1:2,:])
            print('pix_size', pix_size)


            # codec selection
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30

            height, width, channels = img.shape
            video = cv2.VideoWriter(
                os.path.join(output_dir, "video.avi"), fourcc, fps, (width, height)
            )

        outputs = predictor(img)
        outputs = outputs["instances"].to("cpu")

        # filter unwanted detections (only when any detection exists)
        dets = outputs.pred_boxes.tensor.numpy()

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
    if not init:
        video.release()

    # save the final tracks to the json file
    save_json({"tracks": final_tracks}, os.path.join(output_dir, "tracks.json"))

    #process data
    #create_pdf_report(final_tracks, os.path.join(output_dir, "report.pdf"))



#if __name__ == "__main__":
def main_tracker(commandline):
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
    args = parser.parse_args(commandline.split(" "))

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

    #if args.output_dir is not "":
        #CFG["output_dir"] = args.output_dir

    # get the detectron2 configuration and create an output directory
    cfg = get_detectron_cfg()

    if args.output_dir is not "":
        cfg["output_dir"] = args.output_dir

    os.makedirs(cfg.OUTPUT_DIR)

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
