import cv2 as cv
import numpy as np

# Load image
img_bgr = cv.imread('.\images\panda.jpg')
newmask = cv.imread('.\images\panda_mask.jpg',0)

[height,width,channels] = img_bgr.shape
#print(height,width,channels)

# Convert to HSV
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

# Separate channels
img_h = img_hsv[:,:,0]
img_s = img_hsv[:,:,1]
img_v = img_hsv[:,:,2]

mask = np.zeros((height,width),np.uint8)
mask[newmask==0] = 0
mask[newmask==255] = 1

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
cv.grabCut(img_bgr, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

mask3 = np.where((mask==2)|(mask==0),0,255).astype('uint8')
cv.imshow('mask3',mask3)

img_seg = img_bgr*mask2[:,:,np.newaxis]

# Display images
cv.imshow('original',img_bgr)
cv.imshow('image segmentated',img_seg)

cv.waitKey(0)
cv.destroyAllWindows()
