'''
Created on 21.01.2018

@author: oezkan
'''
import Vesselness2D as vs
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

X = np.ones((1024,1024))

img =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/fft.tif',0)
img2 = cv2.imread('C:/Users/oezkan/HasanCan/fft and ROI from andreas/000-filteredImage.tif',0)
ROI = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/201-ErodeMask.tif',0)

img = cv2.equalizeHist(img)

#denoising
img = cv2.bilateralFilter(img,9,75,75)

img_mask=img2*(ROI/255)
kernel = np.ones((5,5),np.uint8)
mask_dil = cv2.erode(ROI,kernel,iterations = 3)/255

start_time1 = time.time()
img_vs = vs.calculateVesselness2D(img_mask, 5)*mask_dil
print(time.time() - start_time1)

plt.subplot(1,2,1),plt.imshow(img,cmap='gray'),plt.title('fft')
plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(img_vs,'gray'),plt.title('vesselness')
plt.xticks([]), plt.yticks([])
plt.colorbar()

plt.show()