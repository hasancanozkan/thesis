'''
Created on 21.01.2018

@author: oezkan
'''
import Vesselness2D as vs
import cv2
import time
from matplotlib import pyplot as plt

img =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/fft.tif',0)
img2 = cv2.imread('C:/Users/oezkan/HasanCan/fft and ROI from andreas/000-filteredImage.tif',0)
ROI = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/201-ErodeMask.tif',0)

img_roi = img2-ROI
#img_roi = img[140:950,230:410]



start_time1 = time.time()
img_vs = vs.calculateVesselness2D(img_roi, 5)
print(time.time() - start_time1)

plt.subplot(1,2,1),plt.imshow(img,cmap='gray'),plt.title('fft')
plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(img_vs,cmap='gray'),plt.title('vesselness')
plt.xticks([]), plt.yticks([])

plt.show()