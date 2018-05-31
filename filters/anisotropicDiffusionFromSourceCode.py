'''
Created on 26.12.2017

@author: oezkan
'''
import cv2
from matplotlib import pyplot as plt
import time
import AnisotropicDiffusionSourceCode as ad
import numpy as np

img_1MB = cv2.imread("originalImages/0000689998.tif",0)
img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_crack2= cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000281_crack.tif",0)
img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_crack3 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000001220_crack.tif",0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006214_bad_.tif',0)

# there are 5 different parameters to change, except option I will have three results for each

kernel = [3,5,10]
for i in range(len(kernel)):
        
    #only changing iterations --> 5,10,20 option1
    start_time1 = time.time()
    img_ad = ad.anisodiff(img_raw, niter=kernel[i],step= (1.,1.), kappa=50,gamma=0.10, option=1)
    print(time.time() - start_time1)
    
    img_ad = np.uint8(img_ad)
    
    
    #option1
    cv2.imwrite('ad'+str(kernel[i])+'.tif',img_ad)
