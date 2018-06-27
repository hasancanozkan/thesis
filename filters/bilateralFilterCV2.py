'''
Created on 15.12.2017

@author: oezkan
'''
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np


img_crack1 = cv2.imread("originalImages/0000331736.tif",0) # 1Mpx 
#img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006214_bad_.tif',0)
img_vessel = cv2.imread('C:/Users/oezkan/Downloads/healthy/01_h.jpg',1)
#img_vessel = np.uint8(img_vessel)
img_vessel_gray = img_vessel
cv2.imwrite('originalGray.tif',img_vessel_gray)

"""
kernel = [3,5,9,11,13,15]
for i in range(len(kernel)):
    start_time1 = time.time()
    img_vessel_bl = cv2.bilateralFilter(img_vessel,kernel[i],100,100)
    print(time.time() - start_time1)
    
    #plt.subplot(1,2,1),plt.imshow(img_vessel,'gray')
    #plt.subplot(1,2,2),plt.imshow(img_vessel_bl,'gray')

    #plt.show()
    
    cv2.imwrite('bl_'+str(kernel[i])+'_rgb.tif',img_vessel_bl)
"""