# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:23:58 2016

@author: uidr9588
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

thresh_min = 60
thresh_max = 80
filename = 'test6.jpg'

image=mpimg.imread(filename)
image_cv2=cv2.imread(filename)
#resized_image = cv2.resize(image, (32, 16)) 
#small = cv2.resize(image, (0,0), fx=0.1, fy=0.1) 
#imgplot = plt.imshow(image)
#plot_resize = plt.imshow(resized_image)

#print(image)


gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) ## using mpimg.imread()
##Find the intensity derivative in the x-direction:
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

##Calculate the absolute value of that derivative:
abssx = np.absolute(sobelx)

##Converting an absolute value image from 64-bit to 8-bit:
scale_factor = np.max(abssx)/255 # Will use this to scale back to 8-bit scale
abssx_scale = (abssx/scale_factor).astype(np.uint8) #rescaling to 8-bit
#
###Create a binary threshold to filter out weak edges:
retval, sxbinary = cv2.threshold(abssx_scale, thresh_min, thresh_max, cv2.THRESH_BINARY)
#
#
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
f.tight_layout()
ax1.imshow(abssx, cmap='gray')
ax1.set_title('1', fontsize=50)
ax2.imshow(sxbinary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#print(abssx)
#print(abssx[0])
#print(abssx[0][200:400])
#print(abssx[0][1100:1220])
histogram = np.sum(abssx[abssx.shape[0]/2:,:], axis=0)
#
print(histogram)

##plot = plt.imshow(abssx)