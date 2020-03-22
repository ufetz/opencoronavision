import cv2 as cv
import numpy as np

# Load image
img_bgr = cv.imread( '.\images\panda.jpg' )
[height,width,channels] = img_bgr.shape

# Load manual mask
mask_man = cv.imread( '.\images\pandaMaskGrabCut1.jpg', 0 )

# Parameter for GrabCut
mask = np.zeros( (height,width), np.uint8 )
bgdModel = np.zeros( (1,65), np.float64 )
fgdModel = np.zeros( (1,65), np.float64 )

# Edit mask
mask[25:298, 140:368] = 3       # probable foreground
mask[mask_man==0] = 0           # background
mask[mask_man==255] = 1         # foreground

# GrabCut with mask
cv.grabCut( img_bgr, mask, None, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK )

# Cut out segmentated panda
mask_gray = np.where( (mask==2)|(mask==0), 0, 255 ).astype('uint8')
img_fgd = cv.bitwise_and( img_bgr, img_bgr, mask=mask_gray )

# Color segmentated areas
mask_bgr = np.zeros( (height,width,3), np.uint8 )
mask_bgr[mask_gray==0] = [0,255,0]
mask_bgr[mask_gray==255] = [0,0,255]
img_seg = cv.addWeighted( img_bgr, 0.75, mask_bgr, 1-0.75, 0 )

# Display images
cv.imshow( 'original image', img_bgr )
cv.imshow( 'image foreground', img_fgd )
cv.imshow( 'segmentated image', img_seg )

cv.waitKey(0)
cv.destroyAllWindows()
