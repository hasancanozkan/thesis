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
img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_nocrack1 = img_nocrack1.astype(np.float32) / 255.0

img_crack1 = img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_crack1 = img_crack1.astype(np.float32) / 255.0

img_crack2 = img_crack2 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000281_crack.tif",0)
img_crack2 = img_crack2.astype(np.float32) / 255.0

img_crack3 = img_crack3 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000001220_crack.tif",0)
img_crack3 = img_crack3.astype(np.float32) / 255.0

start_time1 = time.time()
gf_nocrack1 = guidedFilter(img_nocrack1, img_nocrack1, 4, 0.2**2)
print(time.time() - start_time1)

start_time1 = time.time()
gf_crack1 = guidedFilter(img_crack1, img_crack1, 4, 0.2**2)
print(time.time() - start_time1)

start_time1 = time.time()
gf_crack2 = guidedFilter(img_crack2, img_crack2, 4, 0.2**2)
print(time.time() - start_time1)

start_time1 = time.time()
gf_crack3 = guidedFilter(img_crack3, img_crack3, 4, 0.2**2)
print(time.time() - start_time1)

"""
# Original images
plt.subplot(3,4,1),plt.imshow(img_nocrack1,"gray"),plt.title('Original_nocrack')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,2),plt.imshow(img_crack1,"gray"),plt.title('crack_1')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,3),plt.imshow(img_crack2,"gray"),plt.title('crack_2')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,4),plt.imshow(img_crack3,"gray"),plt.title('crack_3')
plt.xticks([]), plt.yticks([])

#Bilateral Filter
plt.subplot(3,4,5),plt.imshow(gf_nocrack1,"gray"),plt.title('Guided_nocrack')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,6),plt.imshow(gf_crack1,"gray"),plt.title('Guided_crack1')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,7),plt.imshow(gf_crack2,"gray"),plt.title('Guided_crack2')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,8),plt.imshow(gf_crack3,"gray"),plt.title('Guided_crack3')
plt.xticks([]), plt.yticks([])

#Differences
plt.subplot(3,4,9),plt.imshow(img_nocrack1 - gf_nocrack1,"gray"),plt.title('Difference')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,10),plt.imshow(img_crack1 - gf_crack1,"gray"),plt.title('Difference')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,11),plt.imshow(img_crack2 - gf_crack2,"gray"),plt.title('Difference')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,12),plt.imshow(img_crack3 - gf_crack3,"gray"),plt.title('Difference')
plt.xticks([]), plt.yticks([])

plt.show() 
"""

#convert the images back


gf_nocrack1 = np.uint8(gf_nocrack1*255)
gf_crack1 = np.uint8(gf_crack1*255)
gf_crack2 = np.uint8(gf_crack2*255)
gf_crack3 = np.uint8(gf_crack3*255)
    

#save images
cv2.imwrite('gf_nocrack1.tif', gf_nocrack1)
cv2.imwrite('gf_crack1.tif',gf_crack1)
cv2.imwrite('gf_crack2.tif',gf_crack2)
cv2.imwrite('gf_crack3.tif',gf_crack3)

