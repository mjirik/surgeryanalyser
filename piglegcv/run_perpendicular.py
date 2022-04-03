import os
import sys
import math
import cv2
import json
import math
import numpy as np
#from scipy.ndimage import gaussian_filter
from sklearn.cluster import MeanShift
#from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import skeletonize, binary_dilation
from skimage.exposure import histogram
import skimage.color
from skimage.transform import probabilistic_hough_line, resize
from pathlib import Path

from run_report import load_json

import matplotlib.pyplot as plt
from matplotlib import cm

try:
    from incision_detection_mmdet import run_incision_detection
except ImportError:

    from .incision_detection_mmdet import run_incision_detection



def intersectLines( pt1, pt2, ptA, ptB ): 
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
        
        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB;
    dx = xB - x;  dy = yB - y;

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0
    
    ##############
    #found is intersection (xi,yi) in inner segment
    valid = 0
    if x1 != x2:
        if x1 < x2:
            a = x1
            b = x2
        else:
            a = x2
            b = x1
        c = xi
    else:
        #predpoklad, ze pak y jsou ruzne
        if y1 < y2:
            a = y1
            b = y2
        else:
            a = y2
            b = y1
        c = yi
    if (c > a) and (c < b):
        #now second segment
        if x != xB:
            if x < xB:
                a = x
                b = xB
            else:
                a = xB
                b = x
            c = xi
        else:
            #predpoklad, ze pak y jsou ruzne
            if y < yB:
                a = y
                b = yB
            else:
                a = yB
                b = y
            c = yi
        if (c > a) and (c < b):
            valid = 1
   
    
    return ( xi, yi, valid, r, s )



#####################################
def get_frame_to_process(filename):
    if Path(filename).suffix in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
        # image
        img = cv2.imread(filename)
    else:
        ##################
        cap = cv2.VideoCapture(str(filename))
        last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print(last_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)
        ret, img = cap.read()
        i = 0
        while (not ret) and (i < 20):
            print('Last frame capture error, frame', last_frame - i)
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame - i - 1)
            ret, img = cap.read()
            i += 1
        cap.release()
        if not ret:
            print('Last frame capture error')
            return None
        #print(img.shape)
        #plt.imshow(img)
        #plt.show()
        #exit()
        ###################
    return img


def do_incision_detection_by_tracks(img, outputdir, roi, needle_holder_id, canny_sigma):
    # input object tracking data
    json_data = load_json('{}/tracks.json'.format(outputdir))
    sort_data = json_data['tracks'] if 'tracks' in json_data else []
    data_pixel = []
    for frame in sort_data:
        for track_object in frame:
            if (len(track_object) == 5) or ((len(track_object) == 6) and (track_object[5] == needle_holder_id)):
                box = np.array(track_object)
                position = np.array([np.mean([box[0], box[2]]), np.mean([box[1], box[3]])])
                data_pixel.append(position)
                break  # use the first one
    print('Number of data_pixel', len(data_pixel))

    #######################
    # input QR data
    json_data = load_json('{}/qr_data.json'.format(outputdir))
    qr_data = json_data['qr_data'] if 'qr_data' in json_data else {}
    pix_size = qr_data['pix_size'] if 'pix_size' in qr_data else 1.0
    if pix_size == 1.0:
        pix_size = 0.0003  # default 3 desetiny mm na pixel
    is_qr_detected = qr_data['is_detected'] if 'is_detected' in qr_data else False
    print('use pix_size', pix_size)

    ##################
    # compute point as center of processing
    center = np.array([img.shape[1], img.shape[0]]) / 2.0  # default is center of image, (x,y)
    if data_pixel != []:
        data_pixel = np.array(data_pixel)
        track_center = np.median(data_pixel, axis=0)
        if track_center[0] >= 0.0 and track_center[0] < img.shape[1] and track_center[1] >= 0.0 and track_center[1] < \
                img.shape[0]:
            center = track_center  # center is in image
    print('center=', center)
    # plt.plot(center[0], center[1],'o')
    # plt.show()

    #################
    # check and crop image, 1. stage
    column_from = int(center[0] - roi[0] / pix_size / 2.)
    if column_from < 0 or column_from >= img.shape[0]:
        column_from = 0
    column_to = int(center[0] + roi[0] / pix_size / 2.)
    if column_to < 0 or column_to > img.shape[0]:
        column_from = img.shape[0]
    row_from = int(center[1] - roi[1] / pix_size / 2.)
    if row_from < 0 or row_from >= img.shape[1]:
        row_from = 0
    row_to = int(center[1] + roi[1] / pix_size / 2.)
    if row_to < 0 or row_to > img.shape[1]:
        row_from = img.shape[1]
    image = img[row_from:row_to, column_from:column_to, :]
    if image.size == 0:
        print('Image 1. crop is zero')
        return
    image = skimage.color.rgb2gray(image)
    # print(image.shape, column_from, )
    # exit()
    # plt.imshow(image)
    # plt.show()

    edges = canny(image, sigma=canny_sigma)

    edges_location = np.where(edges == 1)
    center[0] += np.median(edges_location[1]) - int(image.shape[1] / 2.0)
    center[1] += np.median(edges_location[0]) - int(image.shape[0] / 2.0)

    ###################################
    # check and crop image, 2. stage
    column_from = int(center[0] - roi[0] / pix_size / 2.)
    if column_from < 0 or column_from >= img.shape[0]:
        column_from = 0
    column_to = int(center[0] + roi[0] / pix_size / 2.)
    if column_to < 0 or column_to > img.shape[0]:
        column_from = img.shape[0]
    row_from = int(center[1] - roi[1] / pix_size / 2.)
    if row_from < 0 or row_from >= img.shape[1]:
        row_from = 0
    row_to = int(center[1] + roi[1] / pix_size / 2.)
    if row_to < 0 or row_to > img.shape[1]:
        row_from = img.shape[1]
    image = img[row_from:row_to, column_from:column_to, :]
    if image.size == 0:
        print('Image 2. crop is zero')
        return
    image = skimage.color.rgb2gray(image[:, :, ::-1])
    # plt.imshow(image)
    # plt.show()
    return image

def main_perpendicular(filename, outputdir, roi=(0.08,0.04), needle_holder_id=0, canny_sigma=2): #(x,y)
    img = get_frame_to_process(filename)


    if img is None:
        print("Input image is None")
        return

    image = do_incision_detection_by_tracks(img, outputdir, roi, needle_holder_id, canny_sigma)


    incision_angle_evaluation(image, canny_sigma, outputdir)

    imgs = run_incision_detection(img, outputdir)
    for i, image in enumerate(imgs):
        incision_angle_evaluation(image, canny_sigma, outputdir, output_filename=f"perpendicular_incision_{i}.jpg")



def incision_angle_evaluation(image, canny_sigma, outputdir, output_filename="perpedicular.jpg"):
    ##  image je oříznutý
    #Resize to uniform size
    image = resize(image, (100,200))
    
    edges = canny(image, sigma=canny_sigma)
    
    #plt.imshow(edges)
    #plt.show()
    #exit()
  
    #thresh = threshold_otsu(image)
    #thresh = threshold_local(image, 101, offset=0)
    
    #dilatation of edges defines amount pixels to detrmine binary Otsu treshold
    edges = binary_dilation(edges)
    edges = binary_dilation(edges)
    edges = binary_dilation(edges)
    edges = binary_dilation(edges)
    edges = binary_dilation(edges)
    
    #his = histogram(image[edges==1], normalize=True)
    #cumhist = np.cumsum(his[0])
    #thresh = 0.5
    #for i in range(len(cumhist)-1):
        #if cumhist[i] < 0.5 and cumhist[i+1] >= 0.5:
            #thresh = his[1][i]
    #plt.plot(his[1], his[0])
    #plt.show()
    #exit()
    #thresh = threshold_otsu(image)
    thresh = threshold_otsu(image[edges==1])
    #edges = skeletonize(image < thresh)
    edges *= image < thresh
    
    #print(thresh)
    #plt.imshow(np.array(edges, dtype=np.int32))
    #plt.show()
    #exit()
    
    
    ############################
    #perpendicular analysis
    tested_angles1 = np.linspace(-np.pi / 10., np.pi / 10., 30, endpoint=False)
    lines1 = probabilistic_hough_line(edges, threshold=10, line_length=25,
                                 line_gap=3, theta=tested_angles1)

    tested_angles2 = np.linspace(np.pi/2.-np.pi/10. , np.pi/2.+np.pi/10., 30, endpoint=False)
    lines2 = probabilistic_hough_line(edges, threshold=10, line_length=75,
                                 line_gap=3, theta=tested_angles2)


    #########
    #plot
    fig = plt.figure()
    fig.suptitle('Perpendicular Analysis', fontsize=14, fontweight='bold')
    plt.imshow(image, cmap=cm.gray)

    ###########
    alphas1 = []
    for line in lines1:
        p0, p1 = line

        dy = p1[1]-p0[1]
        if dy == 0:
            alpha = 0.0
        else:
            if dy < 0:
                dy *= -1
                p = p0
                p0 = p1
                p1 = p

            dx = p0[0]-p1[0]
            alpha = 180.*np.arctan(dx/dy)/np.pi

        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
        #plt.text(p1[0], p1[1],'{:2.1f}'.format(alpha), c='blue', bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 1})
        alphas1.append(alpha)
        
    ##########
    alphas2 = []
    for line in lines2:
        p0, p1 = line

        dx = p0[0]-p1[0]
        if dx == 0:
            alpha = 0.0
        else:
            if dx < 0:
                dx *= -1
                p = p0
                p0 = p1
                p1 = p

            dy = p0[1]-p1[1]
            alpha = 180.*np.arctan(dy/dx)/np.pi

        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
        #plt.text(p1[0], p1[1],'{:2.1f}'.format(alpha), c='red', bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 1})
        alphas2.append(alpha)
        
    ##############
    #analyze alpha for each horizontal and vertical segment
    #print(len(lines1), len(lines2))
    intersections = []
    intersections_alphas = []
    for line2, alpha2 in zip(lines2,alphas2):
        pA, pB = line2
        for line1, alpha1 in zip(lines1, alphas1):
            p0, p1 = line1
            (xi, yi, valid, r, s) = intersectLines(p0, p1, pA, pB)
            if valid == 1:
                #print(xi, yi, r, s)
                #plt.plot(xi, yi, 'o')
                #plt.text(xi, yi,'{:2.1f}'.format(alpha1 - alpha2), c='green', bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 1}, size='large')
                intersections.append([xi, yi])
                intersections_alphas.append(alpha1 - alpha2)
        
    if len(intersections) > 0:
        intersections = np.array(intersections)
        intersections_alphas = np.array(intersections_alphas)
        clustering = MeanShift(bandwidth=3).fit(intersections)    
        #print(clustering.labels_)
        plt.plot(clustering.cluster_centers_[:,0], clustering.cluster_centers_[:,1], 'o')
        for cluster_id in range(clustering.labels_.max()+1):
            #print(cluster_id)
            alpha = np.mean(intersections_alphas[clustering.labels_ == cluster_id])
            plt.text(clustering.cluster_centers_[cluster_id, 0], clustering.cluster_centers_[cluster_id, 1], '{:2.1f}'.format(alpha), c='green', bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 1}, size='large')
        
    plt.axis('off')

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(outputdir, output_filename), dpi=300)
    


      

if __name__ == '__main__':
    main_perpendicular(sys.argv[1], sys.argv[2])
