#!/usr/bin/env python

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import glob
import pickle

import calibrate_camera
import threshold_binary_image
import detect_lanes
import perspective_transform


def superimpose_lane(image, mtx, dist, M, Minv):
    # undistort image
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # get binary image
    ksize = 9
    gradx_r = threshold_binary_image.abs_sobel_rgb_thresh(undist, orient='x', channel='r', sobel_kernel=ksize, thresh=(30, 100))
    s_channel = threshold_binary_image.hls_threshold(undist, channel='s', thresh=(170, 250))
#worked    s_channel = threshold_binary_image.hls_threshold(undist, channel='s', thresh=(170, 255))
#    gradx_s = threshold_binary_image.abs_sobel_hls_thresh(undist, orient='x', channel='s', sobel_kernel=ksize, thresh=(30, 100))
#    s_channel = threshold_binary_image.hls_threshold(undist, channel='s', thresh=(170, 250))
    binary = np.zeros_like(gradx_r)
    binary[(gradx_r == 1) | (s_channel == 1)] = 1

    # perspective_transform
    binary_warped = cv2.warpPerspective(binary, M, dsize=(binary.shape[1], binary.shape[0]), flags=cv2.INTER_LINEAR)

    # detect lane pixels and fit polynomials
    leftx, lefty, rightx, righty = detect_lanes.detect_lane_pixels(binary_warped)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # print('left_fit=', left_fit)
    # print('right_fit=', right_fit)

    # compute radius of curveture in the real world coordinate
    left_curvrad, right_curverad, off_center = detect_lanes.radius_of_curveture_n_offset(leftx, lefty, rightx, righty, image.shape[1])
    #print('left curveture={0:.0f} [m], right curveture={1:.0f} [m]'.format(left_curvrad, right_curverad))

    # Create an image to draw the lines on
    warped_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warped_zero, warped_zero, warped_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, warped_zero.shape[0]-1, warped_zero.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    image_with_lane = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # add radius of curveture text
    curveture_text = 'Radius of curveture ={0:.0f} [m]'.format(left_curvrad)
    cv2.putText(image_with_lane, curveture_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4, cv2.LINE_AA)

    # add offset from center text
    #off_center = detect_lanes.offset_from_center(left_fit, right_fit, binary.shape[1], binary.shape[0])
    off_center_text = 'Vehicle is {0:.2f}m left of center'.format(off_center)
    cv2.putText(image_with_lane, off_center_text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4, cv2.LINE_AA)

    return image_with_lane

if __name__ == '__main__':
    # calibrate camera
    print('Calibratring camera')
    try:
        mtx = pickle.load(open('mtx.pickle', 'rb'))
        dist = pickle.load(open('dist.pickle', 'rb'))
    except (OSError, IOError) as e:
        calib_img_paths = glob.glob('./camera_cal/calibration*.jpg')
        calib_imgs = [ mpimg.imread(x) for x in calib_img_paths ]
        mtx, dist, rvecs, tvecs = calibrate_camera.calibrate_camera(calib_imgs, nx=9, ny=6)
        pickle.dump(mtx, open('mtx.pickle', 'wb'))
        pickle.dump(dist, open('dist.pickle', 'wb'))
    print('Calibration done')

    # get perspective transform matrices
    M, Minv = perspective_transform.getPerspectiveTransformMatrix()

    for image_path in sys.argv[1:]:
        image = mpimg.imread(image_path)
        image_with_lane = superimpose_lane(image, mtx, dist, M, Minv)

        plt.imshow(image_with_lane)

        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        # f.tight_layout()
        # ax1.imshow(image)
        # ax1.set_title('Original image ({})'.format(image_path), fontsize=10)
        # ax2.imshow(image_with_lane)
        # ax2.set_title('With lane', fontsize=10)

        plt.show()
