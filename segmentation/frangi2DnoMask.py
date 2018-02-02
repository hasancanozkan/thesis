'''
Created on Jan 23, 2018

@author: HasanCan
'''
from skimage.filters import frangi
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report
from ROI_function import createROI as roi

MORPH = True

labeled_crack = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/0000689970_fft_label.tif')
img =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/0000689970_fft.tif',0)


# labeled signal 
_, labeled_crack = cv2.threshold(labeled_crack[:,:,2],127,255,cv2.THRESH_BINARY)
kernel_label = np.ones((2,2),np.uint8)
labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)


img = cv2.equalizeHist(img)
img = cv2.bilateralFilter(img,9,75,75) # I can also apply different filters


ROI = roi(img)


img_fr = frangi(img,scale_range=(0.5,4),scale_step=0.5,beta1=0.5,beta2= 0.05)*ROI
kernel_fr = np.ones((5,5),np.uint8)
#img_fr = cv2.erode(img_fr,kernel_fr,iterations =1)
img_fr = cv2.morphologyEx(img_fr, cv2.MORPH_OPEN, kernel_fr)


if MORPH:

    img_fr = img_as_ubyte(img_fr)
    #img_fr.astype(np.float32)
    _, img_thresh = cv2.threshold(img_fr,100,255,cv2.THRESH_BINARY)
    
    # y-axes
    kernel_y = np.ones((10,1),np.uint8)
    img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)
    
    # x-axes
    kernel_x = np.ones((1,10),np.uint8)
    img_morph2 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_x)
    
    img_morph = img_morph1 + img_morph2
    
    contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    defects = []
    for i in range(len(contours)):
        if(cv2.contourArea(contours[i]) > 200):
            defects.append(contours[i])
    
    defectImage = np.zeros((img_thresh.shape))
    cv2.drawContours(defectImage, defects, -1, 1, -1)
    defectImage = img_as_ubyte(defectImage)
    #_, defectImage = cv2.threshold(defectImage,200,255,cv2.THRESH_BINARY) # for now I couldn't realize any help of these
    
#TODO Hough
else:
    print("hough")

print(classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1]))))

plt.subplot(2,2,1)
plt.imshow(img_fr,"gray"),plt.title('frangi')
plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2)
plt.imshow(defectImage,"gray"),plt.title('thresholded and morphed frangi')
plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3)
plt.imshow(labeled_crack,"gray"),plt.title('original label')
plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4)
plt.imshow(np.abs(defectImage-labeled_crack),"gray"),plt.title('difference crack and label')
plt.xticks([]), plt.yticks([])
plt.show()
