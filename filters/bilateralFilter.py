'''
Created on 15.12.2017

@author: oezkan
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

#img = cv2.imread('2 -##-filteredImage.tif',1) # 1 --> gray scale yapti
img = cv2.imread('2 -##-filteredImage.tif',0)
print img.shape

start_time1 = time.time()
blur = cv2.bilateralFilter(img,5,75,75)
print(time.time() - start_time1) 


start_time2 = time.time()
plt.subplot(131),plt.imshow(img,"gray"),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(blur,"gray"),plt.title('Bilateral')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img-blur,"gray"),plt.title('Bilateral')
plt.xticks([]), plt.yticks([])
plt.show()
print(time.time() - start_time2) 