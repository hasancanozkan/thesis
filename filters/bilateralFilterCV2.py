'''
Created on 15.12.2017

@author: oezkan
'''
import cv2
from matplotlib import pyplot as plt
import time


img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_crack2 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000281_crack.tif",0)
img_crack3 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000001220_crack.tif",0)

start_time1 = time.time()
bl_nocrack1 = cv2.bilateralFilter(img_nocrack1,9,75,75)
print(time.time() - start_time1)

start_time1 = time.time()
bl_crack1 = cv2.bilateralFilter(img_crack1,9,75,75)
print(time.time() - start_time1)  

start_time1 = time.time()
bl_crack2 = cv2.bilateralFilter(img_crack2,9,75,75)
print(time.time() - start_time1) 

start_time1 = time.time()
bl_crack3 = cv2.bilateralFilter(img_crack3,9,75,75)
print(time.time() - start_time1) 



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
plt.subplot(3,4,5),plt.imshow(bl_nocrack1,"gray"),plt.title('Bilateral_nocrack')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,6),plt.imshow(bl_crack1,"gray"),plt.title('Bilateral_crack1')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,7),plt.imshow(bl_crack2,"gray"),plt.title('Bilateral_crack2')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,8),plt.imshow(bl_crack3,"gray"),plt.title('Bilateral_crack2')
plt.xticks([]), plt.yticks([])

#Differences
plt.subplot(3,4,9),plt.imshow(img_nocrack1 - bl_nocrack1,"gray"),plt.title('Difference')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,10),plt.imshow(img_crack1 - bl_crack1,"gray"),plt.title('Difference')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,11),plt.imshow(img_crack2 - bl_crack2,"gray"),plt.title('Difference')
plt.xticks([]), plt.yticks([])
plt.subplot(3,4,12),plt.imshow(img_crack3 - bl_crack3,"gray"),plt.title('Difference')
plt.xticks([]), plt.yticks([])

plt.show() 

#save images
cv2.imwrite('bl_nocrack.tif',bl_nocrack1)
cv2.imwrite('bl_crack1.tif',bl_crack1)
cv2.imwrite('bl_crack2.tif',bl_crack2)
cv2.imwrite('bl_crack3.tif',bl_crack3)
