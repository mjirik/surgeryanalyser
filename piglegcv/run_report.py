import os
import io
import sys
import copy
from argparse import ArgumentParser
from typing import Optional

import skimage.color
from loguru import logger
import cv2
import json
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_json(filename):
    if os.path.isfile(filename): 
        with open(filename, 'r') as fr:
            try:
                data = json.load(fr)
            except ValueError as e:
                return {}
            return data
    else:
        return {}


def plot_finger(img, joints, threshold, thickness):
    for i in range(1, len(joints)):
        if (joints[i-1][2] > threshold) and (joints[i][2] > threshold):
            cv2.line(img, (int(joints[i-1][0]), int(joints[i-1][1])), (int(joints[i][0]), int(joints[i][1])), (0, 0, 0), thickness=thickness)
    
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
    #plt.imshow(img)
    #plt.show()


def create_heatmap_report(points:np.ndarray, image:Optional[np.ndarray]=None, filename:Optional[Path]=None):
    """

    :param points: xy points with shape = [i,2]
    :param image: np.ndarray with image
    :param filename: if filename is set the savefig is called and fig is closed
    :return: figure
    """
    # logger.debug(points)
    points = np.asarray(points)
    logger.debug(f"points.shape={points.shape}")
    fig = plt.figure()
    if isinstance(image, np.ndarray):
        im_gray = skimage.color.rgb2gray(image[:, :, ::-1])
        plt.imshow(im_gray, cmap="gray")
    plt.axis("off")

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if points.ndim == 2:
        x, y = points[:, 0], points[:, 1]
        sns.kdeplot(
            x=x, y=y,
            fill=True,
            # thresh=0.1,
            # levels=100,
            # cmap="mako",
            # cmap="jet",
            # palette="jet",
            # cmap="crest",
            cmap="rocket",
            alpha=.5,
            linewidth=0
        )
    else:
        logger.warning("No points found for heatmap")
    if filename is not None:
        # plt.savefig(Path(filename))
        plt.savefig(Path(filename), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    return fig


#ds_threshold [m]
def create_pdf_report(frame_id, data_pixel, image, source_fps, pix_size, QRinit, object_color, object_name, output_file_name, output_file_name2, ds_threshold=0.1, dpi=300):
    
    if frame_id != []:
        data_pixel = np.array(data_pixel)
        data = pix_size * data_pixel
        t = 1.0/source_fps * np.array(frame_id)
        dxy = data[1:] - data[:-1]
        ds = np.sqrt(np.sum(dxy*dxy, axis=1))
        if not QRinit:
            ds_threshold = 200.0

        ds[ds>ds_threshold] = 0.0
        dt = t[1:] - t[:-1]
        t = t[0:-1]
        
        #chech double data
        ind = dt != 0.0
        ds = ds[ind]
        dt = dt[ind]
        t = t[ind]
        
        
        #print(dt)
        L = np.sum(ds)
        T = np.sum(dt)

        fig = plt.figure()
        fig.suptitle(f'Space trajectory analysis of {object_name}', fontsize=14, fontweight='bold')
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        ax.set_title('Plot on the scene image')

        if isinstance(image, np.ndarray):
            ax.imshow(image[:,:,::-1])
        if QRinit:
            box_text = 'Total in-plain track {:.2f} m / {:.2f} sec'.format(L, T)
        else:
            box_text = 'Total in-plain track {:.2f} pix / {:.2f} sec'.format(L, T)
        ax.text(100, 150, box_text, style='italic', bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 10})

        ax.plot(data_pixel[:,0], data_pixel[:,1],'+'+object_color, markersize=12)
        x = data_pixel[0, 0]
        y = data_pixel[0, 1]
        ax.plot(x, y,'go')
        ax.annotate('Start', xy=(x, y), xytext=(x+100, y-100), arrowprops=dict(facecolor='white', shrink=0.001), bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 1})
        x = data_pixel[-1, 0]
        y = data_pixel[-1, 1]
        ax.plot(x, y,'ro')
        ax.annotate('Stop', xy=(x, y), xytext=(x+100, y+100), arrowprops=dict(facecolor='white', shrink=0.001), bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 1})
        ax.axis('off')
        #ax.plot(x[-1], y[-1],'ro')
        #plt.plot(t, dist,'-')

        #plt.show()
        plt.savefig(output_file_name, dpi=dpi)
        print(f'main_report: figures {output_file_name} is saved')

        ##################
        ## second graph
        fig = plt.figure()
        #fig.suptitle('Time analysis', fontsize=14, fontweight='bold')
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        ax.set_title(f'Actual in-plain position of {object_name}')
        ax.set_xlabel('Time [sec]')
        #ax.set_ylabel('Data')
        #ax.plot(t, data[:, 1], "-+r", label="X coordinate [mm]"  )
        #ax.plot(t, data[:, 0], "-+b", label="Y coordinate [m]"  )
        if QRinit:
            track_label = "Track [m]"
            vel_label = "Velocity [m/sec]"
        else:
            track_label = "Track [pix]"
            vel_label = "Velocity [pix/sec]"

        ax.plot(t, np.cumsum(ds), "-"+object_color, label= 'Track', linewidth=3)
        ax.set_ylabel(track_label)
        
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(t, gaussian_filter(ds/dt, sigma=2) , ":"+object_color, label='Velocity', linewidth=3)
        ax2.set_ylabel(vel_label)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax2.legend(loc="upper left")

        #plt.show()
        plt.savefig(output_file_name2, dpi=dpi)
        print(f'main_report: figures {output_file_name2} is saved')
    else:
        print('main_report: No data to report')
    

#####################################
def main_report_old(filename, outputdir, object_colors=["b","r","g","m"], object_names=["Needle holder","Tweezes","Scissors","None"]):
    
    cap = cv2.VideoCapture(filename)
    assert cap.isOpened(), f'Faild to load video file {filename}'

    if cap.isOpened():
        #output video
        video_name = '{}/pigleg_results.avi'.format(outputdir)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(video_name, fourcc, fps, size)

        #input object tracking data
        json_data = load_json('{}/tracks.json'.format(outputdir))
        sort_data = json_data['tracks'] if 'tracks' in json_data else []
   
        #input hand poses data
        json_data = load_json('{}/hand_poses.json'.format(outputdir))
        hand_poses = json_data['hand_poses'] if 'hand_poses' in json_data else []
        
        i = 0
        data_pixels = [[],[],[],[]]
        frame_ids = [[],[],[],[]]
        N = len(sort_data)
        M = len(hand_poses)
        print('Sort data N=', N,' MMpose data M=', M)
        img_first = None
        while (cap.isOpened()):
            flag, img = cap.read()
            if not flag:
                break
            
            #print(i)
            #if i > 500:
                #break

            if img_first is None:
                img_first = img

            #object tracking
            if i < N:
                frame = sort_data[i]
                for track_object in frame:
                    if len(track_object) >= 4:
                        box = np.array(track_object[0:4])
                        position = np.array([np.mean([box[0],box[2]]), np.mean([box[1],box[3]])])

                        if (len(track_object) == 6):
                            class_id = track_object[5]
                        else:
                            class_id = 0
                        if class_id < 4:
                            data_pixels[class_id].append(position)
                            frame_ids[class_id].append(i)

                        ## color
                        color = (0, 255, 0)
                        if class_id == 1:
                            color = (255, 0, 0)
                        if class_id == 2:
                            color = (0, 0, 255)
                        if class_id == 3:
                            color = (0, 255, 255)

                        # draw detection
                        cv2.rectangle(
                            img,
                            (int(box[0]) - 1, int(box[1]) - 1),
                            (int(box[2]) - 1, int(box[3]) - 1),
                            color,
                            thickness=2,
                        )

                        # draw track ID, coordinates: bottom-left
                        cv2.putText(
                            img,
                            str(object_names[class_id]),
                            (int(box[0]) - 2, int(box[3]) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=color,
                            thickness=2,
                        )
            #else:
                #break

            #hand pose tracking
            if i < M:
                if hand_poses[i] != []:
                    plot_skeleton(img, np.asarray(hand_poses[i]), 0.5, 8)

            videoWriter.write(img)

            i += 1

        cap.release()
        videoWriter.release()
        cmd = f"ffmpeg -i {video_name} -ac 2 -y -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {outputdir+'/pigleg_results.mp4'}"
        os.system(cmd)



        #############
        # graph report

        #input QR data
        json_data = load_json('{}/qr_data.json'.format(outputdir))
        qr_data = json_data['qr_data'] if 'qr_data' in json_data else {}
        pix_size = qr_data['pix_size'] if 'pix_size' in qr_data else 1.0
        is_qr_detected = qr_data['is_detected'] if 'is_detected' in qr_data else False

        #plot graphs
        for i, (frame_id, data_pixel, object_color, object_name) in enumerate(zip(frame_ids, data_pixels, object_colors, object_names)):
            create_pdf_report(frame_id, data_pixel, img_first, fps, pix_size, is_qr_detected, object_color,object_name, os.path.join(outputdir, "graph_{}a.jpg".format(i)), os.path.join(outputdir, "graph_{}b.jpg".format(i)))

        print(f'main_report: Video file {filename} is processed!')
    else:
        print(f'main_report: Video file {filename} is not opended!')


#####################
### NEW

def plot3(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


#ds_threshold [m]
def create_video_report(frame_ids, data_pixels, source_fps, pix_size, QRinit, object_colors, object_names,
                        video_size, ds_threshold=0.1, dpi=300, scissors_frames=[]):

    ##################
    ## second graph
    fig = plt.figure(figsize=(video_size[0]/dpi, video_size[1]/dpi), dpi=dpi)

    #fig.suptitle('Time analysis', fontsize=14, fontweight='bold')
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)
    ax.set_title("Object Track Analysis")
    ax.set_xlabel('Time [sec]')
    #ax.set_ylabel('Data')
    #ax.plot(t, data[:, 1], "-+r", label="X coordinate [mm]"  )
    #ax.plot(t, data[:, 0], "-+b", label="Y coordinate [m]"  )
    if QRinit:
        track_label = "Track [m]"
        vel_label = "Velocity [m/sec]"
    else:
        track_label = "Track [pix]"
        vel_label = "Velocity [pix/sec]"
    ax.set_ylabel(track_label)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(vel_label)

    ds_max = 0
    for frame_id, data_pixel, object_color, object_name in zip(frame_ids, data_pixels, object_colors, object_names):
        if frame_id != []:
            data_pixel = np.array(data_pixel)
            data = pix_size * data_pixel
            t = 1.0/source_fps * np.array(frame_id)
            #t_i = 1.0/source_fps * i
            dxy = data[1:] - data[:-1]
            ds = np.sqrt(np.sum(dxy*dxy, axis=1))
            if not QRinit:
                ds_threshold = 200.0

            ds[ds>ds_threshold] = 0.0
            dt = t[1:] - t[:-1]
            t = t[0:-1]

            #chech double data
            ind = dt != 0.0
            ds = ds[ind]
            dt = dt[ind]
            t = t[ind]

            #print(dt)
            L = np.sum(ds)
            T = np.sum(dt)
            ds_cumsum = np.cumsum(ds)
            if len(ds_cumsum) > 0 and ds_cumsum[-1] > ds_max:
                ds_max = ds_cumsum[-1]
            ax.plot(t, ds_cumsum, "-"+object_color, linewidth=1)
            ax2.plot(t, gaussian_filter(ds/dt, sigma=2) , ":"+object_color, label=object_name, linewidth=1)

            print(object_color, object_name)

    # Draw vlines on scissors QR code visible
    t = 1.0 / source_fps * np.array(scissors_frames)
    for frt in t:
        plt.axvline(frt, c="m")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #ax2.legend(loc="upper left")


    logger.debug('main_video_report: OK')
    return fig, ax, ds_max



#####################################
def main_report(filename, outputdir, object_colors=["b","r","g","m"], object_names=["Needle holder","Tweezes","Scissors","None"]):

    cap = cv2.VideoCapture(filename)
    assert cap.isOpened(), f'Faild to load video file {filename}'

    if cap.isOpened():


        #input object tracking data
        json_data = load_json('{}/tracks.json'.format(outputdir))
        sort_data = json_data['tracks'] if 'tracks' in json_data else []

        #input hand poses data
        json_data = load_json('{}/hand_poses.json'.format(outputdir))
        hand_poses = json_data['hand_poses'] if 'hand_poses' in json_data else []

        data_pixels = [[],[],[],[]]
        frame_ids = [[],[],[],[]]
        N = len(sort_data)
        M = len(hand_poses)
        print('Sort data N=', N,' MMpose data M=', M)

        for i, sort_data_i in enumerate(sort_data): #co snimek to jedna polozka, i prazdna []
            frame = sort_data_i
            #print(frame)
            for track_object in frame:
                if len(track_object) >= 4:
                    box = np.array(track_object[0:4])
                    position = np.array([np.mean([box[0],box[2]]), np.mean([box[1],box[3]])])

                    if (len(track_object) == 6):
                        class_id = track_object[5]
                    else:
                        class_id = 0
                    if class_id < 4:
                        data_pixels[class_id].append(position)
                        frame_ids[class_id].append(i)

        #input QR data
        json_data = load_json('{}/qr_data.json'.format(outputdir))
        qr_data = json_data['qr_data'] if 'qr_data' in json_data else {}
        pix_size = qr_data['pix_size'] if 'pix_size' in qr_data else 1.0
        is_qr_detected = qr_data['is_detected'] if 'is_detected' in qr_data else False


        #fig = plt.figure()
        #plt.imshow(im)
        #plt.show()
        #cv2.imshow('aaa', im)
        #cv2.waitkey(0)

        #output video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        size_input_video = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

        size_output_video = size_input_video.copy()
        size_output_video[1] *= 2
        print(size_input_video, size_output_video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = '{}/pigleg_results.avi'.format(outputdir)
        videoWriter = cv2.VideoWriter(video_name, fourcc, fps, size_output_video)

        # frames with visible scissors qr code
        scissors_frames = qr_data["qr_scissors_frames"] if "qr_scissors_frames" in qr_data else []

        fig, ax, ds_max = create_video_report(frame_ids, data_pixels, fps, pix_size, is_qr_detected, object_colors,
                                              object_names, size_input_video, scissors_frames=scissors_frames)

        img_first = None
        i = 0
        while (cap.isOpened()):
            flag, img = cap.read()
            if not flag:
                break

            if not(i % 10):
                logger.debug(f'Frame {i} processed!')

            #if i > 50:
               #break

            if img_first is None:
                img_first = img.copy()

            #object tracking
            if i < N:

                for track_object in sort_data[i]:

                    if len(track_object) >= 4:
                        box = np.array(track_object[0:4])
                        position = np.array([np.mean([box[0],box[2]]), np.mean([box[1],box[3]])])

                        if (len(track_object) == 6):
                            class_id = track_object[5]
                        else:
                            class_id = 0
                        if class_id < 4:
                            ## color
                            color = (255, 255, 255)
                            if object_colors[class_id] == "b":
                                color = (255, 0, 0)
                            if object_colors[class_id] == "r":
                                color = (0, 0, 255)
                            if object_colors[class_id] == "g":
                                color = (0, 255, 0)
                            if object_colors[class_id] == "m":
                                color = (255, 0, 255)

                            # draw detection
                            cv2.circle(
                                img,
                                (int(position[0]), int(position[1])),
                                5,
                                color,
                                thickness=2,
                            )

                            # draw track ID, coordinates: bottom-left
                            cv2.putText(
                                img,
                                str(object_names[class_id]),
                                (int(position[0]+1), int(position[1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=color,
                                thickness=2,
                            )
                #else:
                #break

            #hand pose tracking
            if i < M:
                if hand_poses[i] != []:
                    plot_skeleton(img, np.asarray(hand_poses[i]), 0.5, 8)

            t_i = 1.0/fps * i
            lines = ax.plot([t_i, t_i], [0, ds_max], "-k", label= 'Track', linewidth=1)
            im_graph = plot3(fig)
            im_graph = cv2.cvtColor(im_graph, cv2.COLOR_RGB2BGR) #matplotlib generate RGB channels but cv2 BGR
            ax.lines.pop(-1)
            #print(lines)
            #exit()
            im_graph = im_graph[:,:,:3]
            im = np.concatenate((img, im_graph), axis=0)
            #print(im.shape)
            #exit()
            videoWriter.write(im)

            i += 1

        cap.release()
        videoWriter.release()
        cmd = f"ffmpeg -i {video_name} -ac 2 -y -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {outputdir+'/pigleg_results.mp4'}"
        os.system(cmd)



        #############
        # graph report

        #plot graphs
        for i, (frame_id, data_pixel, object_color, object_name) in enumerate(zip(frame_ids, data_pixels, object_colors, object_names)):
            create_pdf_report(frame_id, data_pixel, img_first, fps, pix_size, is_qr_detected, object_color, object_name, os.path.join(outputdir, "graph_{}a.jpg".format(i)), os.path.join(outputdir, "graph_{}b.jpg".format(i)))
            create_heatmap_report(data_pixel, image=img_first, filename=Path(outputdir) / "heatmap_{}a.jpg".format(i))


        print(f'main_report: Video file {filename} is processed!')
    else:
        print(f'main_report: Video file {filename} is not opended!')


if __name__ == '__main__':
    #main_report('/home/zdenek/mnt/pole/data-ntis/projects/cv/pigleg/detection/plot/data/output.mp4', '/home/zdenek/mnt/pole/data-ntis/projects/cv/pigleg/detection/plot/data/')
    #main_report_old(sys.argv[1], sys.argv[2])
    main_report(sys.argv[1], sys.argv[2])
