#!/usr/bin/env python

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

# returns coordinates of pixels (leftx, lefty, rightx, righty) 
# that belong to left and right lanes
def detect_lane_pixels(binary_warped, out_img=None):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        if out_img is not None:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    if out_img is not None:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    return (leftx, lefty, rightx, righty)


def detect_lanes_pixels2(binary_warped, left_fit, right_fit):

    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return (leftx, lefty, rightx, righty)

ym_per_pix = 30/720   # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension

def radius_of_curveture_n_offset(leftx, lefty, rightx, righty, shape_x):
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    y_eval = np.max(lefty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    left_x = np.poly1d(left_fit_cr)(y_eval*ym_per_pix)
    right_x = np.poly1d(right_fit_cr)(y_eval*ym_per_pix)
    middle_x = shape_x/2 * xm_per_pix
    offset = (left_x + right_x)/2.0 - middle_x
    #print('shape_x={}, left_x={}, right_x={}, middle_x={}'.format(shape_x, left_x, right_x, middle_x))

    return (left_curverad, right_curverad, offset)

if __name__ == '__main__':
    for image_path in sys.argv[1:]:
        image = mpimg.imread(image_path)
        binary_warped = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # print('binary_warped.shape=', binary_warped.shape)

        #
        # first fit
        #

        # get coordinates of lane pixels
        leftx, lefty, rightx, righty = detect_lane_pixels(binary_warped, image)
        print('len(leftx)={}, len(rightx)={}'.format(len(leftx), len(rightx)))

        # Fit a second order polynomial for left and right lane pixels
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        print('left_fit=', left_fit)
        print('right_fit=', right_fit)

        # compute radius of curveture in the real world
        left_curvrad, right_curverad = radius_of_curveture(leftx, lefty, rightx, righty)
        print('left curveture={} [m], right curveture={} [m]'.format(left_curvrad, right_curverad))

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        plt.imshow(image)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()

        #
        # second fit
        #
        # get coordinates of lane pixels based on previous fit
        leftx2, lefty2, rightx2, righty2 = detect_lanes_pixels2(binary_warped, left_fit, right_fit)
        print('len(leftx2)={}, len(rightx2)={}'.format(len(leftx2), len(rightx2)))
        
        # Fit a second order polynomial again
        left_fit2 = np.polyfit(lefty2, leftx2, 2)
        right_fit2 = np.polyfit(righty2, rightx2, 2)
        print('left_fit2=', left_fit2)
        print('right_fit2=', right_fit2)

        # compute radius of curveture in the real world
        left_curvrad2, right_curverad2 = radius_of_curveture(leftx2, lefty2, rightx2, righty2)
        print('left curveture2={} [m], right curveture2={} [m]'.format(left_curvrad2, right_curverad2))

        print('----')
