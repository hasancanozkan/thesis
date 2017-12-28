'''
Created on 26.12.2017

@author: oezkan
'''
import cv2
from matplotlib import pyplot as plt
import time
import AnisotropicDiffusionSourceCode as ad
import numpy as np

img_crack1 = img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_crack2 = img_crack2 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000281_crack.tif",0)
img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_crack3 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000001220_crack.tif",0)

# there are 5 different parameters to change, except option I will have three results for each

"""
#only changing iterations --> 5,10,20 option1
start_time1 = time.time()
ad_crack1_it5_st1_k25_g10_o1 = ad.anisodiff(img_crack1, niter=5,step= (1.,1.), kappa=25,gamma=0.10, option=1)
print(time.time() - start_time1)
start_time1 = time.time()
ad_crack1_it10_st1_k25_g10_o1 = ad.anisodiff(img_crack1, niter=10,step= (1.,1.), kappa=25,gamma=0.10, option=1)
print(time.time() - start_time1)
start_time1 = time.time()
ad_crack1_it20_st1_k25_g10_o1 = ad.anisodiff(img_crack1, niter=20,step= (1.,1.), kappa=25,gamma=0.10, option=1)
print(time.time() - start_time1)

#steps --> 1,2.5,5 with 10 iterations option1
start_time1 = time.time()
ad_crack1_it10_st25_k25_g10_o1 = ad.anisodiff(img_crack1, niter=10,step= (2.5,2.5), kappa=25,gamma=0.10, option=1)
print(time.time() - start_time1)
start_time1 = time.time()
ad_crack1_it10_st5_k25_g10_o1 = ad.anisodiff(img_crack1, niter=10,step= (5.,5.), kappa=25,gamma=0.10, option=1)
print(time.time() - start_time1)
"""
#kappa --> 25,50,100 with 10 iterations and step 1. option1
start_time1 = time.time()
ad_crack1_it10_st1_k50_g10_o1 = ad.anisodiff(img_crack1, niter=10,step= (1.,1.), kappa=50,gamma=0.10, option=1)
print(time.time() - start_time1)
start_time1 = time.time()
ad_crack2_it10_st1_k50_g10_o1 = ad.anisodiff(img_crack2, niter=10,step= (1.,1.), kappa=50,gamma=0.10, option=1)
print(time.time() - start_time1)
start_time1 = time.time()
ad_crack3_it10_st1_k50_g10_o1 = ad.anisodiff(img_crack3, niter=10,step= (1.,1.), kappa=50,gamma=0.10, option=1)
print(time.time() - start_time1)
start_time1 = time.time()
ad_nocrack1_it10_st1_k50_g10_o1 = ad.anisodiff(img_nocrack1, niter=10,step= (1.,1.), kappa=50,gamma=0.10, option=1)
print(time.time() - start_time1)
"""start_time1 = time.time()
ad_crack1_it10_st1_k100_g10_o1 = ad.anisodiff(img_crack1, niter=10,step= (1.,1.), kappa=100,gamma=0.10, option=1)
print(time.time() - start_time1)

#gamma -->0.10 and 0.25 with 10 iterations kappa 50 step 1. option1
start_time1 = time.time()
ad_crack1_it10_st1_k50_g25_o1 = ad.anisodiff(img_crack1, niter=10,step= (1.,1.), kappa=50,gamma=0.25, option=1)
print(time.time() - start_time1)

#####
#only changing iterations --> 5,10,20 option1
start_time1 = time.time()
ad_crack1_it5_st1_k25_g10_o2 = ad.anisodiff(img_crack1, niter=5,step= (1.,1.), kappa=25,gamma=0.10, option=2)
print(time.time() - start_time1)
start_time1 = time.time()
ad_crack1_it10_st1_k25_g10_o2 = ad.anisodiff(img_crack1, niter=10,step= (1.,1.), kappa=25,gamma=0.10, option=2)
print(time.time() - start_time1)
start_time1 = time.time()
ad_crack1_it20_st1_k25_g10_o2 = ad.anisodiff(img_crack1, niter=20,step= (1.,1.), kappa=25,gamma=0.10, option=2)
print(time.time() - start_time1)

#steps --> 1,2.5,5 with 10 iterations option2
start_time1 = time.time()
ad_crack1_it10_st25_k25_g10_o2 = ad.anisodiff(img_crack1, niter=10,step= (2.5,2.5), kappa=25,gamma=0.10, option=2)
print(time.time() - start_time1)
start_time1 = time.time()
ad_crack1_it10_st5_k25_g10_o2 = ad.anisodiff(img_crack1, niter=10,step= (5.,5.), kappa=25,gamma=0.10, option=2)
print(time.time() - start_time1)

#kappa --> 25,50,100 with 10 iterations and step 1. option2
start_time1 = time.time()
ad_crack1_it10_st1_k50_g10_o2 = ad.anisodiff(img_crack1, niter=10,step= (1.,1.), kappa=50,gamma=0.10, option=2)
print(time.time() - start_time1)
start_time1 = time.time()
ad_crack1_it10_st1_k100_g10_o2 = ad.anisodiff(img_crack1, niter=10,step= (1.,1.), kappa=100,gamma=0.10, option=2)
print(time.time() - start_time1)

#gamma -->0.10 and 0.25 with 10 iterations kappa 50 step 1. option2
start_time1 = time.time()
ad_crack1_it10_st1_k50_g25_o2 = ad.anisodiff(img_crack1, niter=10,step= (1.,1.), kappa=50,gamma=0.25, option=2)
print(time.time() - start_time1)




#convert them uint8 -- > because of this tere is a bit lose in accuracy
#option1
ad_crack1_it5_st1_k25_g10_o1 = np.uint8(ad_crack1_it5_st1_k25_g10_o1)
ad_crack1_it10_st1_k25_g10_o1 = np.uint8(ad_crack1_it10_st1_k25_g10_o1)
ad_crack1_it20_st1_k25_g10_o1 = np.uint8(ad_crack1_it20_st1_k25_g10_o1)

ad_crack1_it10_st25_k25_g10_o1 = np.uint8(ad_crack1_it10_st25_k25_g10_o1)
ad_crack1_it10_st5_k25_g10_o1 = np.uint8(ad_crack1_it10_st5_k25_g10_o1)
"""
ad_crack1_it10_st1_k50_g10_o1 = np.uint8(ad_crack1_it10_st1_k50_g10_o1)
ad_crack2_it10_st1_k50_g10_o1 = np.uint8(ad_crack2_it10_st1_k50_g10_o1)
ad_crack3_it10_st1_k50_g10_o1 = np.uint8(ad_crack3_it10_st1_k50_g10_o1)
ad_nocrack1_it10_st1_k50_g10_o1 = np.uint8(ad_nocrack1_it10_st1_k50_g10_o1)
"""
ad_crack1_it10_st1_k100_g10_o1 = np.uint8(ad_crack1_it10_st1_k100_g10_o1)
ad_crack1_it10_st1_k50_g25_o1 = np.uint8(ad_crack1_it10_st1_k50_g25_o1)

#option2
ad_crack1_it5_st1_k25_g10_o2 = np.uint8(ad_crack1_it5_st1_k25_g10_o2)
ad_crack1_it10_st1_k25_g10_o2 = np.uint8(ad_crack1_it10_st1_k25_g10_o2)
ad_crack1_it20_st1_k25_g10_o2 = np.uint8(ad_crack1_it20_st1_k25_g10_o2)

ad_crack1_it10_st25_k25_g10_o2 = np.uint8(ad_crack1_it10_st25_k25_g10_o2)
ad_crack1_it10_st5_k25_g10_o2 = np.uint8(ad_crack1_it10_st5_k25_g10_o2)

ad_crack1_it10_st1_k50_g10_o2 = np.uint8(ad_crack1_it10_st1_k50_g10_o2)
ad_crack1_it10_st1_k100_g10_o2 = np.uint8(ad_crack1_it10_st1_k100_g10_o2)
ad_crack1_it10_st1_k50_g25_o2 = np.uint8(ad_crack1_it10_st1_k50_g25_o2)

#option1
cv2.imwrite('ad_crack1_it5_st1_k25_g10_o1.tif',ad_crack1_it5_st1_k25_g10_o1)
cv2.imwrite('ad_crack1_it10_st1_k25_g10_o1.tif',ad_crack1_it10_st1_k25_g10_o1)
cv2.imwrite('ad_crack1_it20_st1_k25_g10_o1.tif',ad_crack1_it20_st1_k25_g10_o1)
cv2.imwrite('ad_crack1_it10_st25_k25_g10_o1.tif',ad_crack1_it10_st25_k25_g10_o1)
cv2.imwrite('ad_crack1_it10_st5_k25_g10_o1.tif',ad_crack1_it10_st5_k25_g10_o1)
"""
cv2.imwrite('ad_crack1_it10_st1_k50_g10_o1.tif',ad_crack1_it10_st1_k50_g10_o1)
cv2.imwrite('ad_crack2_it10_st1_k50_g10_o1.tif',ad_crack2_it10_st1_k50_g10_o1)
cv2.imwrite('ad_crack3_it10_st1_k50_g10_o1.tif',ad_crack3_it10_st1_k50_g10_o1)
cv2.imwrite('ad_nocrack1_it10_st1_k50_g10_o1.tif',ad_nocrack1_it10_st1_k50_g10_o1)
"""cv2.imwrite('ad_crack1_it10_st1_k100_g10_o1.tif',ad_crack1_it10_st1_k100_g10_o1)
cv2.imwrite('ad_crack1_it10_st1_k50_g25_o1.tif',ad_crack1_it10_st1_k50_g25_o1)
#option2
cv2.imwrite('ad_crack1_it5_st1_k25_g10_o2.tif',ad_crack1_it5_st1_k25_g10_o2)
cv2.imwrite('ad_crack1_it10_st1_k25_g10_o2.tif',ad_crack1_it10_st1_k25_g10_o2)
cv2.imwrite('ad_crack1_it20_st1_k25_g10_o2.tif',ad_crack1_it20_st1_k25_g10_o2)
cv2.imwrite('ad_crack1_it10_st25_k25_g10_o2.tif',ad_crack1_it10_st25_k25_g10_o2)
cv2.imwrite('ad_crack1_it10_st5_k25_g10_o2.tif',ad_crack1_it10_st5_k25_g10_o2)
cv2.imwrite('ad_crack1_it10_st1_k50_g10_o2.tif',ad_crack1_it10_st1_k50_g10_o2)
cv2.imwrite('ad_crack1_it10_st1_k100_g10_o2.tif',ad_crack1_it10_st1_k100_g10_o2)
cv2.imwrite('ad_crack1_it10_st1_k50_g25_o2.tif',ad_crack1_it10_st1_k50_g25_o2)"""