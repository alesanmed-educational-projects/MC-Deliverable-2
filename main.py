# import the necessary packages
import numpy as np
import cv2

from connected_components import find_connected_components
from PIL import Image

image = cv2.imread('resources/bajocontraste.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("Gradient", gradient)
processed_img = Image.fromarray(gradient)
processed_img.save('results/{0}.jpg'.format("Gradient"))

# blur and threshold the image
blurred = cv2.blur(gradient, (3, 3))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
cv2.imshow("Blurred", blurred)
processed_img = Image.fromarray(blurred)
processed_img.save('results/{0}.jpg'.format("Blurred"))

##-------------------------------Lo del entregable 2 va desde aqui-------------------------
cv2.imshow("Thresh", thresh)
processed_img = Image.fromarray(thresh)
processed_img.save('results/{0}.jpg'.format("Thresh"))

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
processed_img = Image.fromarray(closed)
processed_img.save('results/{0}.jpg'.format("Closed"))

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
cv2.imshow("Eroded", closed)
processed_img = Image.fromarray(closed)
processed_img.save('results/{0}.jpg'.format("Eroded"))
closed = cv2.dilate(closed, None, iterations = 4)
cv2.imshow("Dilated", closed)
processed_img = Image.fromarray(closed)
processed_img.save('results/{0}.jpg'.format("Dilated"))

connected = find_connected_components(closed)
print(connected[0])
connected_components = np.uint8((connected[1] * 255) / connected[0])
cv2.imshow("Components Mine", connected_components)
processed_img = Image.fromarray(connected_components)
processed_img.save('results/{0}.jpg'.format("mine_components"))

connected = cv2.connectedComponents(closed)
print(connected[0])
connected_components = np.uint8((connected[1] * 255) / connected[0])
cv2.imshow("Components OpenCV", connected_components)
processed_img = Image.fromarray(connected_components)
processed_img.save('results/{0}.jpg'.format("opencv_components"))

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
_, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
 
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
 
# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
b,g,r = cv2.split(image)
image = cv2.merge([r,g,b])  
processed_img = Image.fromarray(image)
processed_img.save('results/{0}.jpg'.format("Image"))
cv2.waitKey(0)