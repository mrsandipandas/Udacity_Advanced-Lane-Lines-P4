# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:39:05 2016

@author: uidr9588
"""
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

filename = 'test6.jpg'

img=mpimg.imread(filename)
#image_cv2=cv2.imread(filename)
# Convert to HLS color space and separate the S channel
# Note: img is the undistorted image
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]

# Grayscale image
# NOTE: we already saw that standard grayscaling lost color information for the lane lines
# Explore gradients in other colors spaces / color channels to see what might work better
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Sobel x
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Sobel x
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) # Take the derivative in x
abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))


# Threshold x gradient
thresh_min = 50
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# Threshold y gradient
thresh_min_y = 50
thresh_max_y = 100
sxbinaryy = np.zeros_like(scaled_sobely)
sxbinaryy[(scaled_sobely >= thresh_min_y) & (scaled_sobely <= thresh_max_y)] = 1


# Threshold magnitude
mag_thresh_min = 0
mag_thresh_max = 255

magnitude = np.sqrt(sobelx**2 + sobely**2)
scale_factor = np.max(magnitude)/255 # Will use this to scale back to 8-bit scale
magnitude = (magnitude/scale_factor).astype(np.uint8) #rescaling to 8-bit
binary_output = np.zeros_like(magnitude)
binary_output[(magnitude > mag_thresh_min) & (magnitude <= mag_thresh_max)] = 1


# Threshold direction
dir_thresh_min = 0
dir_thresh_max = 255

if sobelx!= 0:
    direction = np.arctan(sobely/sobelx)
    abs_direction = np.absolute(direction)
    binary_output = np.zeros_like(abs_direction)
    binary_output[(abs_direction > dir_thresh_min) & (abs_direction <= dir_thresh_max)] = 1


# Threshold color channel
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1




combined = np.zeros_like(binary_output)
combined[((sxbinary == 1) & (sxbinaryy == 1)) | ((binary_output == 1) & (binary_output == 1))] = 1

# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Stacked thresholds')
ax1.imshow(color_binary)

ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined, cmap='gray')