Advanced Lane Finding Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

The code for this step is implemented as the function `calibrate_camera()` in `calibrate_camera.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in an gray-scaled version of the test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one of the calibration images (`camera_cal/calibration2.jpg`) using the `cv2.undistort()` function and obtained this result: 

<img src="output_images/undistorted.png">

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is how the test image (`test_images/test4.jpg`) looks like after the aformentioned undistortion correction is applied:
<img src="output_images/pipeline_undistorted.png">

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is implemented in `threshold_binary_image.py`.

To generate a binary image, I used the following combination of color and gradient thresholds:
 1. Convert it to an HLS image and then threashold on S values (function: `hls_threshold()`)
 1. Convert it to an HLS and then threashold on the absolute X gradient of its S values (function: `abs_sobel_hls_thresh()`)

Here's an example of my output for this step.

<img src="output_images/threshold_binary_image_out8.jpg">

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The function `getPerspectiveTransformMatrix()` in `perspective_transform.py` implements the perspective transform.

The function calculates a matrix that transforms the following source poitns to destination points with `cv2.getPerspectiveTransform()`.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I took these points from writeup_tempalte.md since they look reasonable for the purpose.

The function also calculates an inverse matrix as well, which will be used later to map fitted lane points back onto the original image.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto the test image (`test_images/straight_lines1.jpg`) and its warped counterpart to verify that the lines appear parallel in the warped image.

<img src="output_images/warped.png">

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The function `detect_lanes()` in `detect_lanes.py` implements an algorithm that detects pixels that comprise left and right lanes. 

The algorithm works roughtly like this:

1. Create a histogram that represents a number of non-zero pixels on each line `x=k` for `k = {0, 1, ... (image_width-1)}` taking into account only pixels in the lower half of the image.
1. Choose the min and max values for the X-axis, `left_begin` and `right_end`, sensively to effectively form a trapezoidal region of interest.
1. Find `x in {left_begin, ...  midpoint-1}` that has the largest number of non-zero pixels. This is the initial value for `leftx_current` for the left lane.
1. Find `x in {midpoint,...right_end-1}` that has the largest number of non-zero pixels. This is the initial value for `rightx_current` for the right lane.

Below I will describe how to detect pixels on the left lane, but exactly the same logic can be applied for the right lane.

1. Record all non-zero pixels (red pixels) in the green rectangle at the bottom (see the picture below). The width of the rectangle is sensively pre-selected as 200 (`margin*2`) and its height as `1/9' of the total height of the image.
1. If more than 50 pixels are found in the box, move the rectangle to the left or right in such a way that the center of the box is aligned to the average X coordinate of those pixels.
1. Move the rectangle up by `1/9` of the height of the image and repeat the process until the green box reaches the top of the image.

Once all lane pixels are found, we can use `numpy.polyfit()` to find a second order polynomial that best fits them, like this:

```python
lefty # y-coordinates of all pixles on the left lane
leftx # x-coordinates of all pixles on the left lane
left_fit = np.polyfit(lefty, leftx, 2)
```
<img src="output_images/lane-pixels-fitted.png">

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function `radius_of_curveture_n_offset()` in `detect_lanes.py` calculates the radius of curvature of the lane as well as the position of the vehicle with respect to center.

The radius of the curvature is calculated as follows:
1. Convert lane pixel coordinates from the pixel space to the world space.
1. Fit a second order polynomial to the coordinates in the world space.
1. Compute the radius of curvature at the bottom of the fitted polynomial with a formula described in this website (this: http://www.intmath.com/applications-differentiation/8-radius-curvature.php)

The position of the vehicle with respect to center is calculated as follows:
1. Calculate the x coordinate of the base of the left lane from the fitted polynomial (`= left_x`)
1. Calculate the x coordinate of the base of the right lane from the fitted polynomial (`= right_x`)
1. From those x coordinates, calculate the x coordinate of the middle of the road (`= (left_x+right_x)/2`)
1. Convert the x coordinate of the middle of the image from the pixel space to the world space (`= middle_x`)
1. The position of the vehicle with respect to center is the different between those coodinates (`= (left_x+right_x)/2 - middle_x`)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


<img src="output_images/with_lane_superimposed.png">

---

### Pipeline (video)

Here's a [link to my video result](./project_video_with_lane.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
