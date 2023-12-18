import os
import io
import sys
import copy
from argparse import ArgumentParser
from typing import Optional

import skimage.color
import skimage.transform
from loguru import logger
import cv2
import json
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy
import scipy.signal
from tools import draw_bbox_into_image
import tools


try:
    from tools import (
        load_json,
        save_json,
        unit_conversion,
        _find_largest_incision_bbox,
        make_bbox_square_and_larger,
        count_points_in_bbox,
        make_bbox_larger,
    )
except ImportError as e:
    from .tools import (
        load_json,
        save_json,
        unit_conversion,
        _find_largest_incision_bbox,
        make_bbox_square_and_larger,
        count_points_in_bbox,
        make_bbox_larger,
    )


def plot_finger(img, joints, threshold, thickness):
    for i in range(1, len(joints)):
        if (joints[i - 1][2] > threshold) and (joints[i][2] > threshold):
            cv2.line(
                img,
                (int(joints[i - 1][0]), int(joints[i - 1][1])),
                (int(joints[i][0]), int(joints[i][1])),
                (0, 0, 0),
                thickness=thickness,
            )

    return img


def plot_skeleton(img, joints, threshold, thickness):
    # right hand
    img = plot_finger(img, joints[0][[0, 1, 2, 3, 4]], threshold, thickness)
    img = plot_finger(img, joints[0][[0, 5, 6, 7, 8]], threshold, thickness)
    img = plot_finger(img, joints[0][[0, 9, 10, 11, 12]], threshold, thickness)
    img = plot_finger(img, joints[0][[0, 13, 14, 15, 16]], threshold, thickness)
    img = plot_finger(img, joints[0][[0, 17, 18, 19, 20]], threshold, thickness)
    # left hand
    img = plot_finger(img, joints[1][[0, 1, 2, 3, 4]], threshold, thickness)
    img = plot_finger(img, joints[1][[0, 5, 6, 7, 8]], threshold, thickness)
    img = plot_finger(img, joints[1][[0, 9, 10, 11, 12]], threshold, thickness)
    img = plot_finger(img, joints[1][[0, 13, 14, 15, 16]], threshold, thickness)
    img = plot_finger(img, joints[1][[0, 17, 18, 19, 20]], threshold, thickness)
    # plt.imshow(img)
    # plt.show()


def calculate_operation_zone_presence(points: np.ndarray, bbox: np.ndarray):

    points = np.asarray(points)
    bbox = np.asarray(bbox)
    #     x, y = points[:, 0], points[:, 1]
    if len(points) > 0:
        return count_points_in_bbox(points, bbox) / float(len(points))
    else:
        return 0


class RelativePresenceInOperatingArea(object):
    def __init__(self):
        self.operating_area_bbox = None

    def set_operation_area_based_on_bboxes(self, bboxes):
        """Create operating area based on largest incision bbox.

        :param bboxes: incision bounding boxes
        :param resize_factor: the factor used for video resize
        :return:
        """
        self.operating_area_bbox = None
        if len(bboxes) > 0:

            # self.operating_area_bbox = _make_bbox_square_and_larger(_find_largest_incision_bbox(bboxes), multiplicator=1.)
            self.operating_area_bbox = make_bbox_larger(
                _find_largest_incision_bbox(bboxes), multiplicator=2.0
            )

    def calculate_presence(self, points):
        if self.operating_area_bbox is not None:
            return calculate_operation_zone_presence(points, self.operating_area_bbox)
        else:
            return 0

    def draw_image(
        self, img: np.ndarray, points: np.ndarray, bbox_linecolor=(0, 255, 128)
    ):
        img = draw_bbox_into_image(
            img, self.operating_area_bbox, linecolor=bbox_linecolor
        )
        points = np.asarray(points)
        bbox = np.asarray(self.operating_area_bbox)
        #     x, y = points[:, 0], points[:, 1]
        if len(points) > 0:
            for point in points:
                if self.operating_area_bbox is None:
                    img = cv2.circle(
                        img,
                        (int(point[0]), int(point[1])),
                        radius=0,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                else:
                    if (
                        point[0] >= bbox[0]
                        and point[0] <= bbox[2]
                        and point[1] >= bbox[1]
                        and point[1] <= bbox[3]
                    ):
                        img = cv2.circle(
                            img,
                            (int(point[0]), int(point[1])),
                            radius=0,
                            color=(0, 255, 0),
                            thickness=2,
                        )
                    else:
                        img = cv2.circle(
                            img,
                            (int(point[0]), int(point[1])),
                            radius=0,
                            color=(0, 0, 255),
                            thickness=2,
                        )
            return img
        else:
            return img


def create_heatmap_report_plt(
    points: np.ndarray,
    image: Optional[np.ndarray] = None,
    filename: Optional[Path] = None,
    bbox: Optional[np.ndarray] = None,
    bbox_linecolor=(128, 255, 0),
):
    """

    :param points: xy points with shape = [i,2]
    :param image: np.ndarray with image
    :param filename: if filename is set the savefig is called and fig is closed
    :param bbox: bounding box to be drawn into image
    :return: figure
    """
    # logger.debug(points)
    points = np.asarray(points)
    logger.debug(f"points.shape={points.shape}")

    if points.ndim != 2:
        logger.warning("No points found for heatmap")
        return None

    fig = plt.figure()
    if isinstance(image, np.ndarray):
        im_gray = skimage.color.rgb2gray(image[:, :, ::-1])
        # one channel gray scale image to 3 channel gray scale image
        im_gray = np.stack([im_gray, im_gray, im_gray], axis=-1)
        if bbox is not None:
            print("bbox")
            print(bbox)
            im_gray = draw_bbox_into_image(im_gray, bbox, linecolor=bbox_linecolor)
        plt.imshow(im_gray, cmap="gray")
    plt.axis("off")

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    x, y = points[:, 0], points[:, 1]
    sns.kdeplot(
        x=x,
        y=y,
        fill=True,
        # thresh=0.1,
        # levels=100,
        # cmap="mako",
        # cmap="jet",
        # palette="jet",
        # cmap="crest",
        cmap="rocket",
        alpha=0.5,
        linewidth=0,
    )
    if filename is not None:
        # plt.savefig(Path(filename))
        plt.savefig(Path(filename), bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    return fig


# ds_threshold [m]
def create_pdf_report(
    frame_id,
    data_pixel,
    image,
    source_fps,
    pix_size,
    QRinit,
    object_color,
    object_name,
    output_file_name,
    output_file_name2,
    ds_threshold=0.1,
    dpi=300,
    visualization_unit="cm",
):
    """

    :param frame_id:
    :param data_pixel:
    :param image:
    :param source_fps:
    :param pix_size:
    :param QRinit:
    :param object_color:
    :param object_name:
    :param output_file_name:
    :param output_file_name2:
    :param ds_threshold:  in [m]
    :param dpi:
    :return:
    """

    if frame_id != []:
        data_pixel = np.array(data_pixel)
        data = pix_size * data_pixel
        t = 1.0 / source_fps * np.array(frame_id)
        dxy = data[1:] - data[:-1]
        ds = np.sqrt(np.sum(dxy * dxy, axis=1))
        if not QRinit:
            ds_threshold = 200.0

        ds[ds > ds_threshold] = 0.0
        ds = unit_conversion(ds, "m", output_unit=visualization_unit)
        dt = t[1:] - t[:-1]
        t = t[0:-1]

        # chech double data
        ind = dt != 0.0
        ds = ds[ind]
        dt = dt[ind]
        t = t[ind]

        # print(dt)
        L = np.sum(ds)
        T = np.sum(dt)
        ds_dt_filtered = gaussian_filter(ds / dt, sigma=2)
        V = np.mean(ds_dt_filtered)

        fig = plt.figure()
        fig.suptitle(
            f"Space trajectory analysis of {object_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        ax.set_title("Plot on the scene image")

        if isinstance(image, np.ndarray):
            ax.imshow(image[:, :, ::-1])

        # check unit
        if QRinit:
            unit = visualization_unit
        else:
            unit = "pix"

        box_text = "Total in-plain track {:.2f} {} / {:.2f} sec".format(L, unit, T)
        ax.text(
            100,
            150,
            box_text,
            style="italic",
            bbox={"facecolor": "white", "alpha": 1.0, "pad": 10},
        )

        ax.plot(data_pixel[:, 0], data_pixel[:, 1], "+" + object_color, markersize=12)
        x = data_pixel[0, 0]
        y = data_pixel[0, 1]
        ax.plot(x, y, "go")
        ax.annotate(
            "Start",
            xy=(x, y),
            xytext=(x + 100, y - 100),
            arrowprops=dict(facecolor="white", shrink=0.001),
            bbox={"facecolor": "white", "alpha": 1.0, "pad": 1},
        )
        x = data_pixel[-1, 0]
        y = data_pixel[-1, 1]
        ax.plot(x, y, "ro")
        ax.annotate(
            "Stop",
            xy=(x, y),
            xytext=(x + 100, y + 100),
            arrowprops=dict(facecolor="white", shrink=0.001),
            bbox={"facecolor": "white", "alpha": 1.0, "pad": 1},
        )
        ax.axis("off")
        # ax.plot(x[-1], y[-1],'ro')
        # plt.plot(t, dist,'-')

        # plt.show()
        plt.savefig(output_file_name, dpi=dpi)
        logger.debug(f"main_report: figures {output_file_name} is saved")

        ##################
        ## second graph
        fig = plt.figure()
        # fig.suptitle('Time analysis', fontsize=14, fontweight='bold')
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        ax.set_title(f"Actual in-plain position of {object_name}")
        ax.set_xlabel("Time [sec]")
        # ax.set_ylabel('Data')
        # ax.plot(t, data[:, 1], "-+r", label="X coordinate [mm]"  )
        # ax.plot(t, data[:, 0], "-+b", label="Y coordinate [m]"  )

        track_label = "Track [{}]".format(unit)
        vel_label = "Velocity [{}/sec]".format(unit)

        ax.plot(t, np.cumsum(ds), "-" + object_color, label="Track", linewidth=3)
        ax.set_ylabel(track_label)

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(t, ds_dt_filtered, ":" + object_color, label="Velocity", linewidth=0.5)
        ax2.set_ylabel(vel_label)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax2.legend(loc="upper left")

        # plt.show()
        plt.savefig(output_file_name2, dpi=dpi)
        logger.debug(f"main_report: figures {output_file_name2} is saved")

        if QRinit:
            L = unit_conversion(L, visualization_unit, "m")
            V = unit_conversion(V, visualization_unit, "m")
            unit = "m"

        return [T, L, V, unit]

    else:
        logger.debug("main_report: No data to report")
        return []


#####################################
# def main_report_old(filename, outputdir, object_colors=["b","r","g","m"], object_names=["Needle holder","Tweezes","Scissors","None"]):
#
#     cap = cv2.VideoCapture(filename)
#     assert cap.isOpened(), f'Faild to load video file {filename}'
#
#     if cap.isOpened():
#         #output video
#         video_name = '{}/pigleg_results.avi'.format(outputdir)
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         videoWriter = cv2.VideoWriter(video_name, fourcc, fps, size)
#
#         #input object tracking data
#         json_data = load_json('{}/tracks.json'.format(outputdir))
#         sort_data = json_data['tracks'] if 'tracks' in json_data else []
#
#         #input hand poses data
#         json_data = load_json('{}/hand_poses.json'.format(outputdir))
#         hand_poses = json_data['hand_poses'] if 'hand_poses' in json_data else []
#
#         i = 0
#         data_pixels = [[],[],[],[]]
#         frame_ids = [[],[],[],[]]
#         N = len(sort_data)
#         M = len(hand_poses)
#         print('Sort data N=', N,' MMpose data M=', M)
#         img_first = None
#         while (cap.isOpened()):
#             flag, img = cap.read()
#             if not flag:
#                 break
#
#             #print(i)
#             #if i > 500:
#                 #break
#
#             if img_first is None:
#                 img_first = img
#
#             #object tracking
#             if i < N:
#                 frame = sort_data[i]
#                 for track_object in frame:
#                     if len(track_object) >= 4:
#                         box = np.array(track_object[0:4])
#                         position = np.array([np.mean([box[0],box[2]]), np.mean([box[1],box[3]])])
#
#                         if (len(track_object) == 6):
#                             class_id = track_object[5]
#                         else:
#                             class_id = 0
#                         if class_id < 4:
#                             data_pixels[class_id].append(position)
#                             frame_ids[class_id].append(i)
#
#                         ## color
#                         color = (0, 255, 0)
#                         if class_id == 1:
#                             color = (255, 0, 0)
#                         if class_id == 2:
#                             color = (0, 0, 255)
#                         if class_id == 3:
#                             color = (0, 255, 255)
#
#                         # draw detection
#                         cv2.rectangle(
#                             img,
#                             (int(box[0]) - 1, int(box[1]) - 1),
#                             (int(box[2]) - 1, int(box[3]) - 1),
#                             color,
#                             thickness=2,
#                         )
#
#                         # draw track ID, coordinates: bottom-left
#                         cv2.putText(
#                             img,
#                             str(object_names[class_id]),
#                             (int(box[0]) - 2, int(box[3]) - 2),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=1,
#                             color=color,
#                             thickness=2,
#                         )
#             #else:
#                 #break
#
#             #hand pose tracking
#             if i < M:
#                 if hand_poses[i] != []:
#                     plot_skeleton(img, np.asarray(hand_poses[i]), 0.5, 8)
#
#             videoWriter.write(img)
#
#             i += 1
#
#         cap.release()
#         videoWriter.release()
#         cmd = f"ffmpeg -i {video_name} -ac 2 -y -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {outputdir+'/pigleg_results.mp4'}"
#         os.system(cmd)
#
#         #############
#         # graph report
#
#         #input QR data
#         json_data = load_json('{}/meta.json'.format(outputdir))
#         qr_data = json_data['qr_data'] if 'qr_data' in json_data else {}
#         pix_size = qr_data['pix_size'] if 'pix_size' in qr_data else 1.0
#         is_qr_detected = qr_data['is_detected'] if 'is_detected' in qr_data else False
#
#         #plot graphs
#         for i, (frame_id, data_pixel, object_color, object_name) in enumerate(zip(frame_ids, data_pixels, object_colors, object_names)):
#             create_pdf_report(frame_id, data_pixel, img_first, fps, pix_size, is_qr_detected, object_color,object_name, os.path.join(outputdir, "graph_{}a.jpg".format(i)), os.path.join(outputdir, "graph_{}b.jpg".format(i)))
#
#         print(f'main_report: Video file {filename} is processed!')
#     else:
#         print(f'main_report: Video file {filename} is not opended!')


#####################
### NEW


def plot3(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


# ds_threshold [m]
def create_video_report_figure(
    frame_ids,
    data_pixels,
    source_fps,
    pix_size,
    QRinit: bool,
    object_colors,
    object_names,
    video_size,
    ds_threshold=0.1,
    dpi=300,
    cut_frames=[],
    visualization_unit="cm",
):

    ##################
    ## second graph
    # fig = plt.figure(figsize=(video_size[0]/dpi, video_size[1]/dpi), dpi=dpi)
    fig = plt.figure()

    # fig.suptitle('Time analysis', fontsize=14, fontweight='bold')
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)
    ax.set_title("Object Track Analysis")
    ax.set_xlabel("Time [sec]")
    # ax.set_ylabel('Data')
    # ax.plot(t, data[:, 1], "-+r", label="X coordinate [mm]"  )
    # ax.plot(t, data[:, 0], "-+b", label="Y coordinate [m]"  )
    if QRinit:
        track_label = f"Track [{visualization_unit}]"
        vel_label = f"Velocity [{visualization_unit}/sec]"
    else:
        track_label = "Track [pix]"
        vel_label = "Velocity [pix/sec]"
    ax.set_ylabel(track_label)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(vel_label)

    ds_max = 0
    for frame_id, data_pixel, object_color, object_name in zip(
        frame_ids, data_pixels, object_colors, object_names
    ):
        if frame_id != []:
            data_pixel = np.array(data_pixel)
            data = pix_size * data_pixel
            t = 1.0 / source_fps * np.array(frame_id)
            # t_i = 1.0/source_fps * i
            dxy = data[1:] - data[:-1]
            ds = np.sqrt(np.sum(dxy * dxy, axis=1))
            if not QRinit:
                ds_threshold = 200.0

            ds[ds > ds_threshold] = 0.0
            ds = unit_conversion(ds, "m", output_unit=visualization_unit)
            dt = t[1:] - t[:-1]
            t = t[0:-1]

            # chech double data
            ind = dt != 0.0
            ds = ds[ind]
            dt = dt[ind]
            t = t[ind]

            # print(dt)
            L = np.sum(ds)
            T = np.sum(dt)
            ds_cumsum = np.cumsum(ds)
            if len(ds_cumsum) > 0 and ds_cumsum[-1] > ds_max:
                ds_max = ds_cumsum[-1]
            ax.plot(t, ds_cumsum, "-" + object_color, linewidth=1)
            ax2.plot(
                t,
                gaussian_filter(ds / dt, sigma=2),
                ":" + object_color,
                label=object_name,
                linewidth=0.2,
            )

            print(object_color, object_name)

    # Draw vlines on scissors QR code visible
    t = 1.0 / source_fps * np.array(cut_frames)
    for frt in t:
        plt.axvline(frt, c="m")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # ax2.legend(loc="upper left")

    logger.debug("main_video_report: OK")
    return fig, ax, ds_max


def _qr_data_processing(json_data: dict, fps):
    """
    Get id of frame with finished knot based on `qr_scissors_frames` key in input dictionary.

    :param qr_data: extract id of cut frames if the key `qr_scissors_frame` exists
    :return:
    """
    qr_data = json_data["qr_data"] if "qr_data" in json_data else {}

    pix_size = qr_data["pix_size"] if "pix_size" in qr_data else 1.0
    is_qr_detected = qr_data["is_detected"] if "is_detected" in qr_data else False
    if ~is_qr_detected:
        pxsz_incision = (
            json_data["pixelsize_m_by_incision_size"]
            if "pixelsize_m_by_incision_size" in json_data
            else None
        )
        if pxsz_incision:
            pix_size = pxsz_incision
            is_qr_detected = True

    # scissors_frames = qr_data["qr_scissors_frames"] if "qr_scissors_frames" in qr_data else []
    # scissors_frames = _scissors_frames(scissors_frames, fps)
    scissors_frames = (
        qr_data["stitch_split_frames"] if "stitch_split_frames" in qr_data else []
    )
    return pix_size, is_qr_detected, scissors_frames


def _scissors_frames(scissors_frames: dict, fps, peak_distance_s=10) -> list:
    """
    Filter scisors frames with minimum peak distance

    :param scissors_frames:
    :param fps:
    :param peak_distance_s:
    :return:
    """
    if len(scissors_frames) == 0:
        return []
    per_frame_data = np.zeros(np.max(scissors_frames) + 1)
    per_frame_data[scissors_frames] = 1

    peak_distance = peak_distance_s * fps
    # scissors_frames = np.asarray(scissors_frames)
    per_frame_data = scipy.ndimage.gaussian_filter(
        per_frame_data, peak_distance, mode="constant", cval=0
    )
    peaks, _ = scipy.signal.find_peaks(per_frame_data, distance=peak_distance)
    # plt.plot(per_frame_data)
    # plt.plot(peaks, per_frame_data[peaks], 'rx')
    # plt.show()
    return peaks.tolist()


def insert_ruler_in_image(img, pixelsize, ruler_size=50, resize_factor=1.0, unit="mm"):
    image_size = np.asarray(img.shape[:2])
    # start_point = np.asarray(image_size) * 0.90
    # start_point = np.array([10,10])
    thickness = int(0.01 * img.shape[0] / resize_factor)
    # start_point = np.array([image_size[1]*0.98, image_size[0]*0.97]) # right down corner
    start_point = np.array([image_size[1] * 0.02, image_size[0] * 0.97])
    ruler_size_px = ruler_size / pixelsize
    end_point = start_point + np.array([ruler_size_px, 0])

    cv2.line(
        img, start_point.astype(int), end_point.astype(int), (255, 255, 255), thickness
    )

    text_point = start_point.astype(np.int) - np.array(
        [0, int(0.020 * img.shape[0]) / resize_factor]
    ).astype(int)
    # img[line]
    text_thickness = int(0.004 * img.shape[0] / resize_factor)
    # logger.debug(f"ruler_size_px={ruler_size_px}")
    # logger.debug(f"text_point={text_point}")
    # logger.debug(f"text_thickness={text_thickness}")
    cv2.putText(
        img,
        f"{ruler_size:0.0f} [{unit}]",
        text_point,
        # (int(position[0]+(circle_radius*2.5)), int(position[1]+circle_radius*0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.0007 * img.shape[0] / resize_factor,
        color=(255, 255, 255),
        thickness=text_thickness,
    )
    return img


#####################################


def bboxes_to_points(outputdir: str, confidence_score_thr: float = 0.0):
    json_data = load_json("{}/tracks.json".format(outputdir))
    sort_data = json_data["tracks"] if "tracks" in json_data else []

    data_pixels = [[] for i in range(15)]
    frame_ids = [[] for i in range(15)]
    N = len(sort_data)
    logger.debug(f"Sort data N={N}")

    for i, sort_data_i in enumerate(
        sort_data
    ):  # co snimek to jedna polozka, i prazdna []
        frame = sort_data_i
        # print(frame)
        for track_object in frame:
            if len(track_object) >= 6:
                box = np.array(track_object[0:4])
                position = np.array(
                    [np.mean([box[0], box[2]]), np.mean([box[1], box[3]])]
                )

                class_id = track_object[5]
                confidence_score = track_object[4]

                if confidence_score > confidence_score_thr:
                    data_pixels[class_id].append(position)
                    frame_ids[class_id].append(i)

    frame_ids_list = np.asarray(frame_ids).tolist()
    data_pixels_list = [
        np.asarray(data_pixels[i]).tolist() for i in range(len(data_pixels))
    ]
    json_metadata = save_json(
        {
            "frame_ids": frame_ids_list,
            "data_pixels_0": np.asarray(data_pixels[0]).tolist(),
            "data_pixels_1": np.asarray(data_pixels[1]).tolist(),
            "data_pixels_2": np.asarray(data_pixels[2]).tolist(),
            "data_pixels_3": np.asarray(data_pixels[3]).tolist(),
            "data_pixels": data_pixels_list,
        },
        "{}/tracks_points.json".format(outputdir),
        update=False,
    )
    return frame_ids, data_pixels, sort_data


def merge_cut_frames(scissors_frames: list, cut_frames: list, fps: float) -> list:
    """Merge list of frames. The frames are merged if they are closer than 10 seconds."""
    scissors_frames = np.asarray(scissors_frames)
    cut_frames = np.asarray(cut_frames)
    all_frames = np.unique(np.concatenate([scissors_frames, cut_frames]))
    all_frames = np.sort(all_frames)
    merged_frames = []
    for frame in all_frames:
        if len(merged_frames) == 0:
            merged_frames.append(frame)
        else:
            if frame - merged_frames[-1] > 10 * fps:
                merged_frames.append(frame)
    return merged_frames


def draw_track_object(
    img,
    box,
    class_id,
    object_name,
    object_color,
    font_scale=0.5,
    thickness=2,
    circle_radius=5,
):

    # 0: Needle holder
    # 1: Forceps
    # 2: Scissors
    # 10: Needle holder bbox
    # 11: Forceps bbox
    # 12: Scissors bbox
    # 13: Left hand bbox
    # 14: Right hand bbox

    ## color
    if object_color == "b":
        color = (255, 0, 0)
    if object_color == "r":
        color = (0, 0, 255)
    if object_color == "g":
        color = (0, 255, 0)
    if object_color == "m":
        color = (255, 0, 255)
    if object_color == "w":
        color = (255, 255, 255)

    color_text = color
    # if confidence_score < confidence_score_thr:
    # color_text = (180, 180, 180)

    # draw detection
    if class_id < 10:  # tips like circles
        position = np.array([np.mean([box[0], box[2]]), np.mean([box[1], box[3]])])
        cv2.circle(
            img,
            (int(position[0]), int(position[1])),
            int(circle_radius),
            color,
            thickness=thickness,
        )
        text_position = (
            int(position[0] + (circle_radius * 2.5)),
            int(position[1] - circle_radius),
        )

        # draw track ID, coordinates: bottom-left
        cv2.putText(
            img,
            str(object_name),
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color_text,
            thickness=thickness,
        )

    if class_id > 12:  # just right and left hand bbbox
        cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            thickness,
        )
        text_position = (int(box[0]), int(box[1]))

        # draw track ID, coordinates: bottom-left
        cv2.putText(
            img,
            str(object_name),
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color_text,
            thickness=thickness,
        )

    return img


def main_report(
    filename,
    outputdir,
    meta: dict,
    object_colors=None,
    # object_colors=["b", "r", "g", "g", "", "", "", "", "", "", "b", "r", "g", "w", "w"],
    # class ID         0               1         2           3        4   5   6  7   8   9     10                   11                12             13                  14
    object_names=None,
    concat_axis=1,
    resize_factor=0.5,
    circle_radius=16.0,
    expected_video_width=1110,
    expected_video_height=420,
    visualization_length_unit="cm",
    confidence_score_thr=0.0,
    oa_bbox_linecolor=[0, 255, 128],
    cut_frames: list = [],
    is_microsurgery: bool = False,
    test_first_seconds: bool = False,
):
    """

    :param filename:
    :param outputdir:
    :param object_colors:
    :param object_names:
    :param concat_axis: axis of original video and graph concatenation. 0 for vertical, 1 for horizontal
    :param cut_frames: list of frames where the stitch cut is detected.
    :return:
    """
    # normální tracking
    # 0: Needle holder
    # 1: Forceps
    # 2: Scissors
    # 10: Needle holder bbox
    # 11: Forceps bbox
    # 12: Scissors bbox
    # 13: Left hand bbox
    # 14: Right hand bbox
    # struktura track boxu: [x1, y1, x2, y2, confidence_score, class_id]

    # microsurgery
    # 0: Needle holder
    # 1: Forceps
    # 2: Forceps curved
    # 3: Scissors
    # struktura track boxu: [x1, y1, x2, y2, confidence_score, class_id]
    if object_colors is None:
        if is_microsurgery:  # udelat lepe, ale jak
            # object_names[2] = "Forceps curved"
            # object_colors[2] = "m"
            object_colors = ["b", "r", "m", "g", "", "", "", "", "", "", "b", "r", "g", "w", "w"],
            object_names = [
                "Needle holder",
                "Forceps",
                "Forceps curved",
                "Scissors",
                "",
                "",
                "",
                "",
                "",
                "",
                "Needle holder bbox",
                "Forceps bbox",
                "Scissors bbox",
                "Left hand bbox",
                "Right hand bbox",
            ],
        else:
            object_colors = ["b", "r", "g", "g", "", "", "", "", "", "", "b", "r", "g", "w", "w"],
            object_names = [
                "Needle holder",
                "Forceps",
                "Scissors",
                "Scissors",
                "",
                "",
                "",
                "",
                "",
                "",
                "Needle holder bbox",
                "Forceps bbox",
                "Scissors bbox",
                "Left hand bbox",
                "Right hand bbox",
            ],

    filename = str(filename)
    outputdir = str(outputdir)

    cap = cv2.VideoCapture(filename)
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert cap.isOpened(), f"Failed to load video file {filename}"

    progress = tools.ProgressPrinter(frame_cnt)
    if cap.isOpened():

        # output video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        size_input_video = [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]

        if concat_axis == 1:
            size_output_fig = [
                int(expected_video_width / 2),
                int(expected_video_height),
            ]
            resize_factor = float(expected_video_height) / float(size_input_video[1])
            size_output_img = [
                int(resize_factor * size_input_video[0]),
                int(expected_video_height),
            ]
            size_output_video = [
                size_output_img[0] + size_output_fig[0],
                int(expected_video_height),
            ]
        else:
            size_output_fig = [
                int(expected_video_width),
                int(expected_video_height / 2),
            ]
            resize_factor = float(expected_video_width) / float(size_input_video[0])
            size_output_img = [
                int(expected_video_width),
                int(resize_factor * size_input_video[1]),
            ]
            size_output_video = [
                int(expected_video_width),
                size_output_img[1] + size_output_fig[1],
            ]

        logger.debug(
            f"size_input_video: {size_input_video}, size_output_video: {size_output_video}, size_output_img: {size_output_img}, resize_factor: {resize_factor}"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video_fn_tmp = Path(f"{outputdir}/pigleg_results.avi")
        output_video_fn = Path(outputdir + "/pigleg_results.mp4")
        videoWriter = cv2.VideoWriter(
            str(output_video_fn_tmp), fourcc, fps, size_output_video
        )

        # input hand poses data
        json_data = load_json("{}/hand_poses.json".format(outputdir))
        hand_poses = json_data["hand_poses"] if "hand_poses" in json_data else []

        M = len(hand_poses)
        logger.debug("MMpose data M=", M)

        # input QR data
        # if meta is None:
        #     meta = load_json('{}/meta.json'.format(outputdir))
        pix_size, is_qr_detected, scissors_frames = _qr_data_processing(meta, fps)
        # scisors_frames - frames with visible scissors qr code
        bboxes = np.asarray(meta["incision_bboxes"])
        relative_presence = RelativePresenceInOperatingArea()
        relative_presence.set_operation_area_based_on_bboxes(bboxes)

        frame_ids, data_pixels, sort_data = bboxes_to_points(
            outputdir, confidence_score_thr
        )
        cut_frames = merge_cut_frames(scissors_frames, cut_frames, fps)

        # just for 4 first objects
        fig, ax, ds_max = create_video_report_figure(
            frame_ids[:4],
            data_pixels[:4],
            fps,
            pix_size,
            is_qr_detected,
            object_colors[:4],
            object_names[:4],
            size_output_fig,
            dpi=300,
            cut_frames=cut_frames,
        )

        img_first = None
        video_frame_first = None
        i = 0
        while cap.isOpened():
            flag, img = cap.read()
            if not flag:
                break

            if img_first is None:
                img_first = img.copy()

            img = skimage.transform.resize(
                img, size_output_img[::-1], preserve_range=True
            ).astype(img.dtype)
            if relative_presence.operating_area_bbox is not None:
                oa_bbox_resized = np.asarray(
                    relative_presence.operating_area_bbox.copy()
                )
                #                 print(oa_bbox_resized)
                oa_bbox_resized = oa_bbox_resized * resize_factor
                img = draw_bbox_into_image(
                    img, oa_bbox_resized, linecolor=oa_bbox_linecolor
                )

            if not (i % 50):
                logger.debug(
                    f"Report on frame {i} done, {progress.get_progress_string(float(i))}"
                )

            if test_first_seconds:
                if i > 100:
                    break

            # object tracking
            if i < len(sort_data):

                for track_object in sort_data[i]:
                    # TODO Zdeněk - vykreslovat bboxy rukou, zatím nevykreslovat bbox nástroje a myslet na to, že to může být mikrochirurgie

                    if len(track_object) >= 6:
                        # struktura track boxu: [x1, y1, x2, y2, confidence_score, class_id]
                        box = (
                            np.array(track_object[0:4]) * resize_factor
                        )  # to size uniform video frame
                        confidence_score = track_object[4]
                        class_id = track_object[5]
                        object_name = object_names[class_id]
                        object_color = object_colors[class_id]

                        img = draw_track_object(
                            img,
                            box,
                            class_id,
                            object_name,
                            object_color,
                            font_scale=0.5 / resize_factor,
                            thickness=int(2.0 / resize_factor),
                            circle_radius=int(circle_radius / resize_factor),
                        )

            # hand pose tracking
            if i < M:
                if hand_poses[i] != []:
                    plot_skeleton(img, np.asarray(hand_poses[i]), 0.5, 8)

            t_i = 1.0 / fps * i
            lines = ax.plot([t_i, t_i], [0, ds_max], "-k", label="Track", linewidth=2)
            im_graph = plot3(fig)
            # fix the video size if it is not correct
            # if not im_graph.shape[:2] == tuple(size_output_frame[::-1]):
            im_graph = skimage.transform.resize(
                im_graph, size_output_fig[::-1], preserve_range=True
            ).astype(img.dtype)
            im_graph = cv2.cvtColor(
                im_graph, cv2.COLOR_RGB2BGR
            )  # matplotlib generate RGB channels but cv2 BGR
            ax.lines.pop(-1)
            # print(lines)
            # exit()
            im_graph = im_graph[:, :, :3]
            if is_qr_detected:
                ruler_size_mm = 5 if meta["is_microsurgery"] else 50
                img = insert_ruler_in_image(
                    img,
                    pixelsize=unit_conversion(pix_size, "m", visualization_length_unit),
                    ruler_size=int(
                        unit_conversion(ruler_size_mm, "mm", visualization_length_unit)
                    ),
                    unit=visualization_length_unit,
                )
            im = np.concatenate((img, im_graph), axis=concat_axis)
            # im = skimage.transform.resize(im, output_shape=[
            # size_output_video[1], size_output_video[0], 3], preserve_range=True).astype(im.dtype)
            # exit()
            if video_frame_first is None:
                video_frame_first = im.copy()
                jpg_pth = Path(outputdir) / "pigleg_results.mp4.jpg"
                if jpg_pth.exists():
                    jpg_pth.unlink()
                cv2.imwrite(str(jpg_pth), video_frame_first)
            videoWriter.write(im)

            i += 1

        video_duration_s = float((i - 1) / fps)
        logger.debug(f"pix_size={pix_size}")
        logger.debug(f"frameshape={im.shape}")
        logger.debug(f"confidence_score_thr={confidence_score_thr}")
        cap.release()
        videoWriter.release()
        cmd = f"ffmpeg -i {str(output_video_fn_tmp)} -ac 2 -y -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {str(output_video_fn)}"
        os.system(cmd)

        #############
        # graph report

        # plot graphs and store statistic
        data_results = {}
        for i, (frame_id, data_pixel, object_color, object_name) in enumerate(
            zip(frame_ids, data_pixels, object_colors, object_names)
        ):
            if frame_id != []:
                simplename = object_name.lower().strip().replace(" ", "_")
                # print(cut_frames)

                frame_idx_start = 0
                frame_idx_stop = len(frame_id)
                object_full_name = f"{object_name}"
                stitch_name = "all"
                j_before = 0
                for cut_id, cut_frame in enumerate([0] + cut_frames):

                    if cut_id > 0:
                        object_full_name = f"{object_name} stitch {cut_id}"
                        stitch_name = f"stitch_{cut_id}"
                        for j, frame in enumerate(frame_id):
                            if frame > cut_frame:
                                frame_idx_start = j_before
                                frame_idx_stop = j
                                j_before = j
                                break
                    print(
                        object_full_name,
                        " frame_idx_start:frame_idx_stop",
                        frame_idx_start,
                        frame_idx_stop,
                    )

                    res = create_pdf_report(
                        frame_id[frame_idx_start:frame_idx_stop],
                        data_pixel[frame_idx_start:frame_idx_stop],
                        img_first,
                        fps,
                        pix_size,
                        is_qr_detected,
                        object_color,
                        object_name,
                        os.path.join(
                            outputdir,
                            f"graph_{i}c_{simplename}_trajectory_{stitch_name}.jpg",
                        ),
                        os.path.join(
                            outputdir, f"fig_{i}a_{simplename}_graph_{stitch_name}.jpg"
                        ),
                    )

                    oz_presence = relative_presence.calculate_presence(
                        data_pixel[frame_idx_start:frame_idx_stop]
                    )
                    image_presence = relative_presence.draw_image(
                        img_first.copy(),
                        data_pixel[frame_idx_start:frame_idx_stop],
                        bbox_linecolor=oa_bbox_linecolor,
                    )
                    cv2.imwrite(
                        str(Path(outputdir) / f"{simplename}_area_presence.jpg"),
                        image_presence,
                    )
                    # obj_name = object_name.lower().replace(" ", "_")
                    #
                    #
                    if len(res) > 0:
                        [T, L, V, unit] = res
                        # data_results[object_name] = {}
                        data_results[f"{object_full_name} length [{unit}]"] = L
                        data_results[f"{object_full_name} visibility [s]"] = T
                        data_results[f"{object_full_name} velocity"] = V
                        data_results[f"{object_full_name} unit"] = unit
                        data_results[f"{object_full_name} visibility [%]"] = float(
                            100 * T / video_duration_s
                        )
                        data_results[f"{object_full_name} area presence [%]"] = float(
                            100 * oz_presence
                        )

                    oa_bbox = None
                    if simplename == "needle_holder":
                        logger.debug("adding operating area to the heatmap")
                        oa_bbox = relative_presence.operating_area_bbox

                    create_heatmap_report_plt(
                        data_pixel[frame_idx_start:frame_idx_stop],
                        image=img_first,
                        filename=Path(outputdir)
                        / f"fig_{i}b_{simplename}_heatmap_{stitch_name}.jpg",
                        bbox=oa_bbox,
                        bbox_linecolor=oa_bbox_linecolor,
                    )

        # save statistic to file
        # save_json(data_results, os.path.join(outputdir, "results.json"))

        # TODO unlink but wait for finishing ffmpeg
        if output_video_fn.exists():
            output_video_fn_tmp.unlink()

        print(f"main_report: Video file {filename} is processed!")
        return data_results
    else:
        print(f"main_report: Video file {filename} is not opended!")
        return {}

    ##save perpendicular
    # data_results = load_json(os.path.join(outputdir, "results.json"))
    # perpendicular_data = load_json(os.path.join(outputdir, "perpendicular.json"))


if __name__ == "__main__":
    # main_report('/home/zdenek/mnt/pole/data-ntis/projects/cv/pigleg/detection/plot/data/output.mp4', '/home/zdenek/mnt/pole/data-ntis/projects/cv/pigleg/detection/plot/data/')
    # main_report_old(sys.argv[1], sys.argv[2])
    # main_report(sys.argv[1], sys.argv[2])
    main_report(sys.argv[1], sys.argv[2], concat_axis=1, confidence_score_thr=0.7)
