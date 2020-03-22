import cv2 as cv
import numpy as np

# Load image
img_bgr = cv.imread('.\images\panda.jpg')
newmask = cv.imread('.\images\panda_mask3.jpg',0)

[height,width,channels] = img_bgr.shape

# Set parameter for grabCut function
mask = np.zeros((height,width),np.uint8)
mask[25:298, 140:368] = 3
mask[newmask==0] = 0
mask[newmask==255] = 1
rect = (140,25,228,273)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

cv.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK)
#cv.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,255).astype('uint8')
img_seg = cv.bitwise_and(img_bgr, img_bgr, mask=mask2)

# Display images
cv.imshow('original',img_bgr)
cv.imshow('newmask',newmask)
cv.imshow('mask',mask2)
cv.imshow('image segmentated',img_seg)

cv.imwrite('.\images\pandaSegGrabCut1.jpg',img_seg)

cv.waitKey(0)
cv.destroyAllWindows()
