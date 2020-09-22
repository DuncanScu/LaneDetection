#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys


#The lane detection pipeline follows the dollowing steps:
#1 Pre-process image using grayscale and gaussian blur
#2 Apply canny edge detection to the image
#3 Apply masking region to the image
#4 Apply Hough transform to the image
#5 Extrapolate the lines found in the hough transform to construct the left and right lane lines
#6 Add the extrapolated lines to the input image

#1 Pre-Processing
def grayscale(img): #converts the image to grayscale which is needed for canny analysis
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image = mpimg.imread('test_images/solidYellowCurve2.jpeg')

# grayscale the image
grayscaled = grayscale(image)
plt.imshow(grayscaled, cmap='gray')
plt.show()

#Applying Gaussian Smoothing Function to the image
#Averaging out anomalous gradient changes within the image

def gaussian_blur(img, kernel_size): #Applies a gaussian Noise Kernal, which blurs the images to remove detail and noise
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

kernelSize = 5
gaussianBlur = gaussian_blur(grayscaled, kernelSize)

#2 Canny Edge Detection
#Operator which uses the horizontal and vertical gradients of the pixel values of an image
#It is a multi stage algorithm with the following stages: Noise Reduction, Finding intenstiy gradient of an image, non-maximum suppression, hysteresis thresholding

def canny(img, low_threshold, high_threshold): #Applies the transform
    return cv2.Canny(img, low_threshold, high_threshold)

#canny
minThreshold = 100
maxThreshold = 200
edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)
plt.imshow(edgeDetectedImage)
plt.show()

#3 Applying masking regions
#This gets rid of areas that are not of interst, as only the two lanes in immediate view are required
#These can be filtered out by making a polygon region of interest and removing pixels that are not in the polygon

def region_of_interest(img, vertices): #Applies an image mask
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with 
    #depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#apply mask
lowerLeftPoint = [130, 540]
upperLeftPoint = [410, 350]
upperRightPoint = [570, 350]
lowerRightPoint = [915, 540]

pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
lowerRightPoint]], dtype=np.int32)
masked_image = region_of_interest(edgeDetectedImage, pts)

plt.imshow(masked_image)
plt.show()

#4 Hough Transform
#Now the edges have been detected, and a region of interest has been designated
#We want to identify lones which indicated lane lines via the hough transform
#The transofrm converts a "x vs. y" line to a pint in "gradient vs. intercept" space
#Points within the image correspond to lines in hough space
#an intersection in space will correspond to a line in cartesian space
#We can use this to find lines from the pixel outputs

def hough_lines(img, rho, theta, threshold, min_line_len, min_line_gap): #Img is the output of the canny transform, returning an image with hough lines drawn
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
              minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    
    draw_lines(line_img, lines)
    return line_img
#!!!!Following is commented out to allow for section 5!!!
#def draw_lines(img, lines, color=[255, 0, 0], thickness=2): #Function to draw the lines with colour and thickness
    #for line in lines:
        #for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2): #Draws lines with color and
    imshape = img.shape
    
    # these variables represent the y-axis coordinates to which 
    # the line will be extrapolated to
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    
    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []
    
    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient, intercept = np.polyfit((x1,x2), (y1,y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)
            
            if (gradient > 0):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]
    
    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)
    
    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)
    
    # Make sure we have some points in each lane line category
    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(img, (upper_left_x, ymin_global), 
                      (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global), 
                      (lower_right_x, ymax_global), color, thickness)        
#hough variables
rho = 1
theta = np.pi/180
threshold = 30
min_line_len = 20 
max_line_gap = 20

houged = hough_lines(masked_image, rho, theta, 
                  threshold, min_line_len, max_line_gap)

plt.imshow(houged)
plt.show()

#5 Extrapolate the individual small lines and construct left and right lanes
#The hough transform delivered small lines based upon interestions in hte hough space, which is now used to construct lanes
#This is done by seperating the small lines into two groups. One with a positive gradient and the other negative
#The lanes slant towards each other due to the camera angle creating the opposing gradients
#Taking advantage of gradients and intersepts means global lane lines are identified
#The lines are extrapolated to the edge detected  pixel with min y axis cordinate and the pixel with the max y axis coordinate
#!!!Section 5 code is inserted above.....see line 109


#6 Adding the extrapolated lines back into the image
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)
  
# outline the input image
colored_image = weighted_img(houged, image)
plt.imshow(colored_image)
plt.show()