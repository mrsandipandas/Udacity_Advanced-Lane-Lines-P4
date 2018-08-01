# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1]: ./results/draw_chess_points.png "Chess board points"
[image2]: ./results/undistorted.png "Undistorted"
[image3]: ./results/undis_example.png "Undistorted frame"
[image4]: ./results/abs_sobel_hsl.png "Combining thresholds"
[image5]: ./results/warped.png "Warped"
[image6]: ./results/histogram.png "Histogram"
[image7]: ./results/polinomy.png "Polynomial"

Initially, I started off with the calibration and then moved on to the individual frame and finally to the pipeline for the video processing.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image?
The code for this step is contained in the first code cell of the IPython notebook located in "./Udacity-Advanced-Lane-Lines/calibration.ipynb".


I started by reading all the calibration images given for the calibration of the camera. I applied the function ***cv2.findChessboardCorners*** to each image to detect and obtain the coordinates of the chessboard corners, this function will return 2 values one boolean (retval), True if the corners were detected and a list(corners) . If the coordinates were detected I would append this coordinates to my list called `imgpoints` which is a list to save the points corresponding to chessboard corners on the image in a 2D plane. In order to be able to map the position of the all the corners found in each image a list called `objpoints` this list contains a so called "object points" that have the form of (x,y,z), where z is always zero, this object points represent the coordinates of the chesboard cornes in the real world. 

In order to make sure that the corners were detected correctly I used the function ***cv2.drawChessboardCorners*** to draw the corner found on top of the image tested. This is the result.

![alt text][image1]

I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the ***cv2.calibrateCamera()*** function.  I applied this distortion correction to the test image using the ***cv2.undistort()*** function and obtained this result: 

![alt text][image2]


### Pipeline (single images)

#### Has the distortion correction been correctly applied to each image?
In order to show the results obtained once the calibration was obtained, I will show you the distortion correction aplied in one of the test images. However, in the IPython notebook located in "./Udacity-Advanced-Lane-Lines/calibration.ipynb" all the test images are undistorted. You can also find the undistorted images in the "./Udacity-Advanced-Lane-Lines/undistortion" folder.

The function ***cv2.undistort(img, mtx, dist, None, mtx)*** takes an image, the calibration matrix, the distortion coefficients; `mtx` and `dist` are obtained from the calibration of the camera. The function returns an image without distortion. 

![alt text][image3]


#### Has a binary image been created using color transforms, gradients or other methods?
In order to filter the images and keep only the lane lines, I used the absolute value of Sobel x since this will help me to keep the vertical lines and it will dimish the horizontal lines. This is helpful since the lane lines I'm looking for are vertical. Additilonally I also implemented a color thresholding using the HLS color space and applying a threshold on the S(saturation) channel since this will work for yellow and white lines.


![alt text][image4]



#### Has a perspective transform been applied to rectify the image?

The function `warp` can be found in the IPython notebook called `Development` on the 10th cell. It takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  

I hardcode the source and destination points in the following manner:

```
    image_size = image.shape
    
    horizon = np.uint(2*image_size[0]/3)
    bottom = np.uint(image_size[0])
    center_lane = np.uint(image_size[1]/2)
    offset = 20

    x_left_bottom = 0
    x_right_bottom = 2*center_lane
    x_right_upper = center_lane +  center_lane/4.5
    x_left_upper = center_lane + offset - center_lane/5


    source = np.float32([[x_left_bottom,bottom],[x_right_bottom,bottom],[x_right_upper,horizon],[x_left_upper,horizon]])

    destination = np.float32([[0,image_size[0]],[image_size[1],image_size[0]], [image_size[1],0],[0,0]])

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]


#### Have lane line pixels been identified in the rectified image and fit with a polynomial?

After calculating the warped image. First I calculate the histogram across the image, this was implemented using `numpy.sum()` to sum all the pixels on the rows, the result is the following:

![alt text][image6]

Then from the histogram, each of the peaks are calculated which seves as a starting point for the lines. This code can be found on the IPython notebook called `Development` on cell #16, then I implemented a function called `sliding_window` to iterate through the image and find the x coordinates and y coordinates of the pixels that corresponded to the lanes starting from the centers that I found earlier.

Once I got the pixels I fit them to a 2nd order polynomial of the form:
				`f(y)=Ay^2 + By + C ` 

The result obtained for one of the test images is the following:

![alt text][image7]


#### Having identified the lane lines, has the radius of curvature of the road been estimated? And the position of the vehicle with respect to center in the lane?

Yes. I used the x and y coordinates obtained with the polynomial to calculate the curvature of the lane. This code can be found on the IPython notebook called `LaneDetection` using function `calculate_curvature_pixels` using the formula as described [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php).

Also, depending upon my prespective transform, I have chosen the following:

`xm_per_pix = 3.7/900` and `ym_per_pix = 17/720` for more accuracy in the determing the position of the car. Also, instead of using the last polynomial, I averaged as follows to get much more smoother result:

`lanes_middle_distance = abs(right_lane.recent_xfitted[:][-1].mean() + left_lane.recent_xfitted[:][-1].mean())/2`

Also, depending on the position of the vehicle, I have added whether the car is to the left or to the right of the center in the video results to provide for undestanding of the result. 

---

### Pipeline (video)

#### Does the pipeline established with the test images work to process the video?

The code used for the final pipeline can be found on the IPython notebook called `LaneDetection`, this notebook will contain all the function described above but without the images demostrations. 

Aditionally contains the code for the Line class that was implemented to keep track of the lanes in each frame as well as other helper functions. 

Here's a [link to my video result](https://youtu.be/pGj49kfwW-g)

<a href="http://www.youtube.com/watch?feature=player_embedded&v=pGj49kfwW-g " target="_blank">
	<img src="http://img.youtube.com/vi/pGj49kfwW-g/0.jpg" alt="Video result of advanced lane finding" width="640" height="480" border="10"/>
</a>

The link to the challenge video is [here](https://youtu.be/ORrcnVYa6d8) and the harder challenge results can be seen [here](https://youtu.be/bD9xLTo91ks).

---

## Discussion

  
This project was very interesting and challenging. I learnt a lot of technics used for computer vision like measuring and correcting distortion, perpective trasnformation, how to use diffent color spaces to filter images, etc. One of the most challenging parts for me was how to establish the right parameters to determine which detected line was not a good one so sometimes I added lanes that messed up the average of the lane. So definitely I should work on a better strategy or on a more sophisticated method to identified bad lines. I also could explore more on the threshold and color spaces to find a better filter. Also, in the harder challenge as understood the lines are curving in both the directions too frequently. So, the neighbouring area search strategy does not work and the searching starts from scratch. Also, a better strategy can be to check the tangents of the lane lines at similar y position and move forward if they match. At the same time it would also be good to check the uniformity of the distance between the lines in the unwarped image. These kind of strategies can improve the performance and accurany of detected lines under varios conditions.

