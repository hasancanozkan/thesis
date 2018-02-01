'''
Created on 01.02.2018

@author: oezkan
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import img_as_ubyte



#img = cv2.imread('C:/Users/oezkan/HasanCan/fft and ROI from andreas/000-filteredImage.tif',0)
img =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/0000720002_fft.tif',0)

img = cv2.equalizeHist(img)
img = cv2.bilateralFilter(img,9,75,75)

_,ROI = cv2.threshold(img,40,255,cv2.THRESH_BINARY)
#kernel_ROI = np.ones((2,2),np.uint8)
#ROI = cv2.erode(ROI,kernel_ROI,iterations = 1) # this can be erode a well


kernel_x = np.ones((1,8),np.uint8)
img_morph2 = cv2.morphologyEx(ROI, cv2.MORPH_OPEN, kernel_x)
    
    
kernel_y = np.ones((8,1),np.uint8)
img_morph1 = cv2.morphologyEx(ROI, cv2.MORPH_OPEN, kernel_y)

img_morph = img_morph2 * img_morph1 

contours,_ = cv2.findContours(ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



defects = []
for i in range(len(contours)):
        if(cv2.contourArea(contours[i]) > 300):
            defects.append(contours[i])
            
                        
defectImage = np.zeros((ROI.shape))
cv2.drawContours(defectImage, defects, -1, 1, -1)

  
    

plt.subplot(131),plt.imshow(img_morph2,"gray"),plt.title('img_morph2')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_morph1,"gray"),plt.title('img_morph1')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(defectImage,"gray"),plt.title('img_morph')
plt.xticks([]), plt.yticks([])
plt.show()


