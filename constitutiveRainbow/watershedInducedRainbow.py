import cv2
import numpy as np
from skimage import measure, color, io
from skimage.segmentation import clear_border

imagePath = '/Volumes/Aortas/Aorta/preprocessedImages/inducedRainbow/thresholded/orange/10-30/C5RB-051/220908_C5RB-051_10-30_descending_stitch_orange-1.tif'

img1 = cv2.imread(imagePath)
img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#blurr if necessary
#blurred = cv2.GaussianBlur(redImage, (5, 5), 0)

#Otsu is not always preferred to be used
ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)

#Morphological operations to remove small noise - opening
#To remove holes we can use closing
#not good for induced model because the "noise" is mostly signal


opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# let us start by identifying sure background area
# dilating pixels a few times increases cell boundary to background.
# This way whatever is remaining for sure will be background.
#The area in between sure background and foreground is our ambiguous area.
#Watershed should find this area for us.
sure_bg = cv2.dilate(thresh,kernel,iterations=10)
#normally use opening

# Finding sure foreground area using distance transform and thresholding
#intensities of the points inside the foreground regions are changed to
#distance their respective distances from the closest 0 value (boundary).
#https://www.tutorialspoint.com/opencv/opencv_distance_transformation.html

opening8Bit = thresh.astype(np.uint8)
dist_transform = cv2.distanceTransform(opening8Bit,cv2.DIST_L2,3)
#normally use opening

#Let us threshold the dist transform by starting at 1/2 its max value.
#print(dist_transform.max()) gives about 21.9
ret2, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)

# Unknown ambiguous region is nothing but bkground - foreground
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret3, markers = cv2.connectedComponents(sure_fg)
#sure_fg8bit = sure_fg.astype(np.uint8)
#ret3, markers = cv2.connectedComponents(sure_fg8bit)

#test how many markers are there
#np.unique(markers)

markers = markers+10
markers[unknown==255] = 0
markers = cv2.watershed(img1,markers)

#Let us color boundaries in yellow. OpenCv assigns boundaries to -1 after watershed.
img1[markers == -1] = [0,255,255]
img2 = color.label2rgb(markers, bg_label=0)

print("New round 0.1")

cv2.imshow('Overlay on original image', img1)
cv2.imshow('Colored Grains', img2)
cv2.imshow('Sure FG', sure_fg)
cv2.waitKey(0)
