'''
Created on 15.12.2017

@author: oezkan
'''
import cv2
from matplotlib import pyplot as plt
import time


img_crack1 = cv2.imread("originalImages/0000331736.tif",0) # 1Mpx 
#img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)


start_time1 = time.time()
bl_crack1 = cv2.bilateralFilter(img_crack1,9,75,75)
print(time.time() - start_time1)



cv2.imwrite('bl_crack1Mpx.tif',bl_crack1)
