#!/usr/bin/env python

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import glob
import pickle
from moviepy.editor import VideoFileClip

from calibrate_camera import calibrate_camera
from perspective_transform import getPerspectiveTransformMatrix
from superimpose_lane import superimpose_lane 

if __name__ == '__main__':
    # calibrate camera
    print('Calibratring camera')
    try:
        mtx = pickle.load(open('mtx.pickle', 'rb'))
        dist = pickle.load(open('dist.pickle', 'rb'))
    except (OSError, IOError) as e:
        calib_img_paths = glob.glob('./camera_cal/calibration*.jpg')
        calib_imgs = [ mpimg.imread(x) for x in calib_img_paths ]
        mtx, dist, rvecs, tvecs = calibrate_camera(calib_imgs, nx=9, ny=6)
        pickle.dump(mtx, open('mtx.pickle', 'wb'))
        pickle.dump(dist, open('dist.pickle', 'wb'))
    print('Camera calibration done')

    # get perspective transform matrices
    M, Minv = getPerspectiveTransformMatrix()

    input_path = sys.argv[1]
    output_path = input_path.replace('.mp4', '_with_lanes.mp4')

    def process_image(image):
        return superimpose_lane(image, mtx, dist, M, Minv)

    VideoFileClip(input_path).fl_image(process_image).write_videofile(output_path, audio=False)
