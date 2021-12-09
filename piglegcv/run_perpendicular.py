import os
import sys
from argparse import ArgumentParser
import cv2
import json
import numpy as np
#from scipy.ndimage import gaussian_filter
import math
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

import skimage.color
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt
from matplotlib import cm


#####################################
def main_perpendicular(filename, outputdir):
    

    #input object tracking data
    sort_data = []
    with open('{}/tracks.json'.format(outputdir), "r") as fr:
        data = json.load(fr)
        sort_data = data['tracks']
    
    data_pixel = []
    for frame in sort_data:
        if frame != []:
            box = np.array(frame[0])
            position = np.array([np.mean([box[0],box[2]]), np.mean([box[1],box[3]])])
            data_pixel.append(position)

                 
    data_pixel = np.array(data_pixel)
    center = np.mean(data_pixel, axis=0)
    
    img = cv2.imread(filename)
    #plt.plot(data_pixel[:,0], data_pixel[:,1], 'rx')
    #plt.plot(center[0], center[1], 'o')
    #plt.imshow(img[:,:,::-1])
    #plt.show()
    
    img = img[630:770,760:1050,:]
    #plt.imshow(img[:,:,::-1])
    #plt.show()
    
    image = skimage.color.rgb2gray(img)
    print(np.max(image), np.min(image))
    
    #edges = canny(image,  sigma=1)
    thresh = threshold_otsu(image)
    #print('thresh', thresh)
    #edges = (image < thresh)
    edges = skeletonize(image < thresh)
    
    #theta = np.array([-np.pi/10.,np.pi/10.])
    tested_angles = np.linspace(-np.pi / 10., np.pi / 10., 20, endpoint=False)
    lines = probabilistic_hough_line(edges, threshold=5, line_length=15,
                                 line_gap=5, theta=tested_angles)

    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
    
    tested_angles = np.linspace(np.pi/2.-np.pi/10. , np.pi/2.+np.pi/10., 10, endpoint=False)
    lines = probabilistic_hough_line(edges, threshold=20, line_length=45,
                                 line_gap=3, theta=tested_angles)

    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
    
    exit()
    
    
    
    
    edges = canny(image,  sigma=1)
    #edges = image < 0.25
    #edges = filters.sobel(image)
    #edges = 1.0 - image
    #print(edges)
    #exit()
    plt.imshow(image)
    plt.figure()

    tested_angles = np.linspace(-np.pi / 2 - np.pi/8, -np.pi / 2 + np.pi/8, 250, endpoint=False)

    h, theta, d = hough_line(edges, theta=tested_angles)

    #Generating figure 1
    #fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 6))
    #ax = axes.ravel()

    plt.imshow(edges, cmap=cm.gray)
    #a.set_title('Input image')
    #ax1.set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
                np.rad2deg(theta[-1] + angle_step),
                d[-1] + d_step, d[0] - d_step]
    plt.imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 5.5)
    plt.show()
    exit()
    #ax2.set_title('Hough transform')
    #ax2.set_xlabel('Angles (degrees)')
    #ax2.set_ylabel('Distance (pixels)')
    #ax2.axis('image')

    #ax3.imshow(image, cmap=cm.gray)
    #ax3.set_ylim((image.shape[0], 0))
    #ax3.set_axis_off()
    #ax3.set_title('Detected lines')
    #for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=30, num_peaks=10, min_angle=20)):
        #(x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #ax3.axes.axline((x0, y0), slope=np.tan(angle + np.pi/2))


    #tested_angles2 = np.linspace(- np.pi/8, np.pi/8, 250, endpoint=False)
    #h, theta, d = hough_line(edges, theta=tested_angles2)
    #for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=1, num_peaks=15, min_angle=20)):
        #(x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #ax3.axes.axline((x0, y0), slope=np.tan(angle + np.pi/2),c='r')


    plt.imshow(image, cmap=cm.gray)
    #plt.set_ylim((image.shape[0], 0))
    #set_axis_off()
    #set_title('Detected lines')
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=30, num_peaks=10, min_angle=20)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        plt.Axes.axline((x0, y0), slope=np.tan(angle + np.pi/2))


    tested_angles2 = np.linspace(- np.pi/8, np.pi/8, 250, endpoint=False)
    h, theta, d = hough_line(edges, theta=tested_angles2)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=1, num_peaks=15, min_angle=20)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        plt.Axes.axline((x0, y0), slope=np.tan(angle + np.pi/2),c='r')


    plt.tight_layout()
    plt.show()
      

if __name__ == '__main__':
    main_perpendicular(sys.argv[1], sys.argv[2])
