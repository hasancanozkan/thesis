'''
Created on 22.12.2017

@author: oezkan
'''
import cv2
from cv2.ximgproc import guidedFilter
from matplotlib import pyplot as plt
import time
import numpy as np


# Load image as grayscale image and convert it to
# a float32 scale with grayvalues ranging from [0,1]
#img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
#img_crack1 = cv2.imread("originalImages/0000331736.tif",0) # 1Mpx 
img_crack1 = img_crack1.astype(np.float32) / 255.0


start_time1 = time.time()
gf_crack1 = guidedFilter(img_crack1, img_crack1, 4, 0.2**2)
print(time.time() - start_time1)

#convert the images back
gf_crack1 = np.uint8(gf_crack1*255)
#save images
cv2.imwrite('gf_crack1Mpx01.tif', gf_crack1)