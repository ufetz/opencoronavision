import cv2 as cv
import numpy as np

img = cv.imread('.\images\panda_masked.jpg')
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

img_h = img_hsv[:,:,0]
img_s = img_hsv[:,:,1]
img_v = img_hsv[:,:,2]

ret, hue_bin = cv.threshold(img_h, 165, 255, cv.THRESH_BINARY)
ret, sat_bin = cv.threshold(img_s, 170, 255, cv.THRESH_BINARY)

mask = cv.bitwise_and(hue_bin,sat_bin)
cv.imwrite('.\images\panda_mask.jpg',mask)

cv.imshow('masked image',img)
#cv.imshow('hue binary',hue_bin)
#cv.imshow('saturation binary',sat_bin)
cv.imshow('mask',mask)

cv.waitKey(0)
cv.destroyAllWindows()
