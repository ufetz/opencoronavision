import cv2
import numpy as np

# Load image
img_bgr = cv2.imread('.\images\panda.jpg')

# Convert to HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Separate channels
img_h1 = img_hsv[:,:,0]
img_s1 = img_hsv[:,:,1]
img_v1 = img_hsv[:,:,2]

# Threshold
ret1, bin_sat = cv2.threshold(img_s1, 21, 255, cv2.THRESH_BINARY_INV)
ret2, bin_hue = cv2.threshold(img_h1, 100, 255, cv2.THRESH_BINARY)

# Morphological opening
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
bin_sat_op = cv2.morphologyEx(bin_sat, cv2.MORPH_OPEN, kernel1)
bin_hue_op = cv2.morphologyEx(bin_hue, cv2.MORPH_OPEN, kernel1)

# Mask
mask1 = cv2.bitwise_or(bin_sat,bin_hue)
mask2 = cv2.bitwise_or(bin_sat_op,bin_hue_op)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
mask3 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel2)

# Bear
bear1 = cv2.bitwise_and(img_bgr,img_bgr, mask=mask1)
bear3 = cv2.bitwise_and(img_bgr,img_bgr, mask=mask3)

# Display images
cv2.imshow('original',img_bgr)

cv2.imshow('hue HSV',img_h1)
cv2.imshow('saturation HSV',img_s1)
cv2.imshow('value/intensity HSV',img_v1)

#cv2.imshow('hue HLS',img_h2)
#cv2.imshow('luminance HLS',img_l2)
#cv2.imshow('saturation HLS',img_s2)

cv2.imshow('saturation binary',bin_sat)
cv2.imshow('hue binary',bin_hue)
cv2.imshow('binary saturation opening',bin_sat_op)
cv2.imshow('binary hue opening',bin_hue_op)
cv2.imshow('mask1',mask1)
cv2.imshow('mask2',mask2)
cv2.imshow('mask3',mask3)

cv2.imshow('bear 1',bear1)
cv2.imshow('bear 3',bear3)

cv2.waitKey(0)
cv2.destroyAllWindows()
