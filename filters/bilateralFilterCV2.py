'''
Created on 15.12.2017

@author: oezkan
'''
import cv2
from matplotlib import pyplot as plt
import time


img_crack1 = cv2.imread("originalImages/0000331736.tif",0) # 1Mpx 
#img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006214_bad_.tif',0)

kernel = [3,5,7,9]
for i in range(len(kernel)):
    start_time1 = time.time()
    bl_crack1 = cv2.bilateralFilter(img_raw,kernel[i],100,100)
    print(time.time() - start_time1)
    
    
    
    cv2.imwrite('bl_'+str(kernel[i])+'.tif',bl_crack1)
