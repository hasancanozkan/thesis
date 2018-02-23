'''
Created on Jan 23, 2018

@author: HasanCan
'''
from adaptedFrangi import frangi
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report
from ROI_function import createROI as roi
from fftFunction import fft
from cv2.ximgproc import guidedFilter
from AnisotropicDiffusionSourceCode import anisodiff as ad
import pywt
from Vesselness2D import calculateVesselness2D,getHighestVesselness
import math


img_raw =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000689998.tif',0)
img_raw2=  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736.tif',0)
img_raw3 = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006604_bad_.tif',0)

mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
labeled_crack = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000689998_CrackLabel.tif')
labeled_crack2 = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736_CrackLabel.tif')
labeled_crack3 = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000006604_bad_.tif')

maskRoi = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000689998_BusLabel.tif',0) 


_,maskRoi = cv2.threshold(maskRoi,253,255,cv2.THRESH_BINARY)



_, labeled_crack = cv2.threshold(labeled_crack3[:,:,2],127,255,cv2.THRESH_BINARY)
plt.imshow(labeled_crack,'gray'),plt.show()
kernel_label = np.ones((2,2),np.uint8)
labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)

# apply fft for grid fingers
img_fft = fft(img_raw3, mask_fft)

# apply histogram equalization
img_Eq = cv2.equalizeHist(img_fft)

img_roi=roi(img_Eq)


img_filtered = cv2.bilateralFilter(img_Eq,9,75,75)

''' param_scale i arttirdim
'''
param_scale = [(1.5,1.6),(2.0,2.1),(2.5,2.6),(3.0,3.6),(3.5,3.6),(4.0,4.1),(4.5,4.6),(5.0,5.1)]

v = [[],[],[],[],[],[],[],[]]

for k in range(len(param_scale)):
    v[k] = frangi(img_filtered,scale_range=param_scale[k],scale_step=1,beta1=0.5,beta2= 0.05)
    v[k] = img_as_ubyte(v[k])

min_img = np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])*img_roi
max_img  = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])
#_, min_img = cv2.threshold(min_img,15,255,cv2.THRESH_BINARY)

img_fr = frangi(img_filtered,scale_range=(2,2.1),scale_step=0.5,beta1=0.5,beta2= 0.05)*img_roi
_,img_fr_thres = cv2.threshold(img_fr,5,255,cv2.THRESH_BINARY)

img_fr = img_as_ubyte(img_fr)

#img_fr.astype(np.float32)
_, img_thresh = cv2.threshold(min_img,20,255,cv2.THRESH_BINARY)
     
    # y-axes
kernel_y = np.ones((10,1),np.uint8)
img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)
     
    # x-axes
kernel_x = np.ones((1,10),np.uint8)
img_morph2 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_x)
  
img_morph = img_morph1 + img_morph2
    
_,contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
defects = []
for i in range(len(contours)):
    if(cv2.contourArea(contours[i]) > 200):
        defects.append(contours[i])
     
defectImage = np.zeros((img_thresh.shape))
cv2.drawContours(defectImage, defects, -1, 1, -1)
defectImage = img_as_ubyte(defectImage)

print classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1])))


plt.subplot(2,4,1),plt.imshow(labeled_crack3,'gray'),plt.title('labeled')
plt.subplot(2,4,2),plt.imshow(min_img,'gray'),plt.title('min_img')
plt.subplot(2,4,3),plt.imshow(defectImage,'gray'),plt.title('defectImage')
plt.subplot(2,4,4),plt.imshow(img_fr,'gray'),plt.title('img_fr')


plt.show()

