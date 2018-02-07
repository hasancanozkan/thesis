'''
Created on 31.01.2018

@author: oezkan
'''
from skimage.filters import frangi
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report
from ROI_function import createROI as roi
from filters.fftFunction import fft

#labeled_crack = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/crack_label.tif')
#img2 = cv2.imread('C:/Users/oezkan/HasanCan/fft and ROI from andreas/000-filteredImage.tif',0)
#ROI = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/201-ErodeMask.tif',0)
img =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736.tif',0)
mask = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)


img_fft = fft(img, mask)




"""
defect_image = roi(img)
defect_image = img_as_ubyte(defect_image)


#edges_def = cv2.Canny(defect_image,100,200)

lines = cv2.HoughLines(defect_image,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),10)
"""
"""
_, labeled_crack = cv2.threshold(labeled_crack[:,:,2],127,255,cv2.THRESH_BINARY)
kernel_label = np.ones((2,2),np.uint8)
labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =2)

img = cv2.equalizeHist(img)
img = cv2.bilateralFilter(img,25,75,75)

kernel_ROI = np.ones((5,5),np.uint8)
ROI = cv2.erode(ROI,kernel_ROI,iterations = 3)


img_fr = frangi(img,scale_range=(0.5,4),scale_step=0.5,beta1=0.5,beta2= 0.05)*(ROI/255)
kernel_fr = np.ones((2,2),np.uint8)
img_fr2 = cv2.erode(img_fr,kernel_fr,iterations =2)

img_fr3 = img_as_ubyte(img_fr2)
_, img_thresh = cv2.threshold(img_fr3,120,255,cv2.THRESH_BINARY)


kernel_y = np.ones((10,1),np.uint8)
img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)

contours,_ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

defects = []
for i in range(len(contours)):
        if(cv2.contourArea(contours[i]) > 200):
            defects.append(contours[i])
            
                        
defectImage = np.zeros((img_thresh.shape))
cv2.drawContours(defectImage, defects, -1, 1, -1)
defectImage = img_as_ubyte(defectImage)
_, defectImage = cv2.threshold(defectImage,120,255,cv2.THRESH_BINARY)
"""    
    

plt.subplot(121),plt.imshow(img,"gray"),plt.title('img')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_fft,"gray"),plt.title('fft')
plt.xticks([]), plt.yticks([])
plt.show()


