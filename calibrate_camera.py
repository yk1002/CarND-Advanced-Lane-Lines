#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def calibrate_camera(rgb_chessboard_imgs, nx, ny):
    img_points = [] # 2D point in image plane
    obj_points = [] # 3D points in real world space

    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # x, y cordinates
    
    for rgb_img in rgb_chessboard_imgs:
        # convert image to grayscale
        gs_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(gs_img, (nx, ny), None)
        if found:
            img_points.append(corners)
            obj_points.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gs_img.shape[::-1], None, None)
    assert ret

    return (mtx, dist, rvecs, tvecs)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('Usage: {} nx ny calib_img_path1 [calib_img_path2..] test_img_path'.format(sys.argv[0]))
        sys.exit(0)
    
    nx, ny = int(sys.argv[1]), int(sys.argv[2])
    calib_img_paths = sys.argv[3:-1]
    test_img_path = sys.argv[-1]

    print('nx=', nx)
    print('ny=', ny)
    print('calib_img_paths=', calib_img_paths)
    print('test_img_path=', test_img_path)

    calib_imgs = [ mpimg.imread(x) for x in calib_img_paths ]
    mtx, dist, rvecs, tvecs =  calibrate_camera(calib_imgs, nx, ny)
    
    test_img = mpimg.imread(test_img_path)
    undist_test_img = cv2.undistort(test_img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
#    f, (ax1, ax2) = plt.subplots(1, 2)
    f.tight_layout()
    ax1.imshow(test_img)
    ax1.set_title('Original ({})'.format(test_img_path), fontsize=10)
    ax2.imshow(undist_test_img)
    ax2.set_title('Undistorted', fontsize=10)
#    plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.0)

    plt.show()
