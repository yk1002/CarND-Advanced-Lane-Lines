#!/usr/bin/env python

import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import pickle
import cv2
import scipy.misc

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1 
    return binary_output

def abs_sobel_hls_thresh(img, orient='x', channel='s', sobel_kernel=3, thresh=(20, 100)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    channel = hls[:,:,{'h':0, 'l':1, 's':2}[channel]]
    if orient == 'x':
        sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1 
    return binary_output

def abs_sobel_rgb_thresh(img, orient='x', channel='r', sobel_kernel=3, thresh=(20, 100)):
    channel = img[:,:,{'r':0, 'g':1, 'b':2}[channel]]
    if orient == 'x':
        sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1 
    return binary_output
    
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def hls_threshold(img, channel='s', thresh=(170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    channel = hls[:,:,{'h':0, 'l':1, 's':2}[channel]]
    binary_output = np.zeros_like(channel)
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

if __name__ == '__main__':
    save_result = False
    save_index = 0

    for image_path in sys.argv[1:]:

        # Read in an image and grayscale it
        image = mpimg.imread(image_path)

        # Choose a Sobel kernel size
        ksize = 9 # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
        gradx_s = abs_sobel_hls_thresh(image, orient='x', channel='s', sobel_kernel=ksize, thresh=(30, 100))
        gradx_r = abs_sobel_rgb_thresh(image, orient='x', channel='r', sobel_kernel=ksize, thresh=(30, 100))
        s_channel = hls_threshold(image, channel='s', thresh=(170, 250))

        combined = np.zeros_like(gradx_s)
        combined[(gradx_s == 1) | (s_channel == 1)] = 1

        if save_result:
            save_index += 1
            combined_save = np.dstack((combined, combined, combined))*255
            scipy.misc.imsave('threshold_binary_image_out{}.jpg'.format(save_index), combined_save)

        # Plot the result
        #f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))
#        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image ({})'.format(image_path), fontsize=10)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Combined Thresholding', fontsize=10)
        #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.show()
