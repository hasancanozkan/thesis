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
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006214_bad_.tif',0)
img_vessel = cv2.imread('C:/Users/oezkan/Downloads/healthy/01_h.jpg',0)

#plt.imshow(img_vessel)

plt.show()
img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
#img_crack1 = cv2.imread("originalImages/0000331736.tif",0) # 1Mpx 
img_raw = img_raw.astype(np.float32) / 255.0
img_vessel = img_vessel.astype(np.float32) / 255.0
kernel = [1,2,4,8]
for i in range(len(kernel)):
        
    start_time1 = time.time()
    img_guided = guidedFilter(img_vessel, img_vessel,kernel[i] , 0.2**2)
    
    print(time.time() - start_time1)
    #plt.subplot(1,2,1),plt.imshow(img_vessel,'gray')
    #plt.subplot(1,2,2),plt.imshow(img_guided,'gray')

    #plt.show()
    #convert the images back
    img_guided = np.uint8(img_guided*255)
    #save images
    cv2.imwrite('gf_'+str(kernel[i])+'.tif', img_guided)