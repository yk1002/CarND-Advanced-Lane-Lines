#!/usr/bin/env python
import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import scipy.misc

# coordinates taken from writepup_template.md
src_points = np.float32([[585, 460], [203, 720],  [1127, 720], [695, 460]])
dst_points = np.float32([[320,   0], [320, 720],  [960,  720], [960, 0  ]])

def getPerspectiveTransformMatrix():
    return  (cv2.getPerspectiveTransform(src_points, dst_points), cv2.getPerspectiveTransform(dst_points, src_points))

if __name__ == '__main__':
    save_result = True
    save_index = 0

    M, Minv = getPerspectiveTransformMatrix()

    for image_path in sys.argv[1:]:
        image = mpimg.imread(image_path)
        cv2.polylines(image, [np.int32(src_points)], True, (255,0,0), 2)
        warped = cv2.warpPerspective(image, M, dsize=(image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

        if save_result:
            save_index += 1
            scipy.misc.imsave('perspective_transform_out{}.jpg'.format(save_index), warped)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original image with source points ({})'.format(image_path), fontsize=10)
        ax2.imshow(warped)
        ax2.set_title('Warped', fontsize=10)
        #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.show()
