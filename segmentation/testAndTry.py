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
from fftFunction import fft
from cv2.ximgproc import guidedFilter
from AnisotropicDiffusionSourceCode import anisodiff as ad
import pywt
from Vesselness2D import calculateVesselness2D,getHighestVesselness


img_raw =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000689998.tif',0)
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
labeled_crack = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/fftResults/0000689998_fft_label.tif')
#labeled_crack2 = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000006604_bad_.tif',0)
#img_raw2 = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006604_bad_.tif',0)



#img_filtered = cv2.bilateralFilter(img_raw,9,75,75) # I can also apply different filters
#img_raw = img_raw.astype(np.float32) / 255.0
#img_filtered = guidedFilter(img_raw, img_raw, 4, 0.8**2)
#img_raw = np.uint8(img_raw*255)
#img_filtered = np.uint8(img_filtered*255)

# apply fft for grid fingers

img_fft = fft(img_raw, mask_fft)
img_Eq = cv2.equalizeHist(img_fft)
img_Eq = img_Eq.astype(np.float32)/255
ROI = roi(img_fft)
plt.imshow(ROI,'gray')
plt.colorbar()
plt.show()


img_filtered = ad(img_Eq, niter=30,step= (19.,19.), kappa=100,gamma=0.25, option=1)
plt.imshow(img_filtered,'gray')
plt.title('ani')
plt.colorbar()
plt.show()
#img_fft = cv2.bilateralFilter(img_fft,9,75,75) # I can also apply different filters
# labeled signal 
_, labeled_crack = cv2.threshold(labeled_crack[:,:,2],127,255,cv2.THRESH_BINARY)
kernel_label = np.ones((2,2),np.uint8)
labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)

img_fr1 = frangi(img_filtered,scale_range=(0.1,0.2),scale_step=1,beta1=0.5,beta2= 0.001,  black_ridges=True)
img_fr2 = frangi(img_filtered,scale_range=(1,1.1),scale_step=1,beta1=0.5,beta2= 0.001,  black_ridges=True)
min_img = np.minimum(img_fr1,img_fr2)
#img_fr2 = calculateVesselness2D(img_fft, 3, 0.5, 4)*ROI
plt.subplot(1,3,1)
plt.imshow(img_fr1,"gray"),plt.title('img_fr')
plt.subplot(1,3,2)
plt.imshow(img_fr2,"gray"),plt.title('img_fr2')
plt.subplot(1,3,3)
plt.imshow(min_img,"gray"),plt.title('img_fr3')
plt.show()



# TODO
# MASK the image (BUSBARS CANCELING)
# FFT of the image
# Equalize Hist

# load labeled image to compare with

#list_of_parameters: 
# wavelet:
# bilateral: k= (5,5) to (15x15) stepsize +2 | sigma_p, 25-150 (+25) | simga_g = 25-150 (+25)
# guided:    r= 2 - 7 to  stepsize +1 | epsilon (+0.1 - 0.4) (stepsize 0.1)
# anistropic 

# frangi

for i in range (0,4,1):
    if (i == 0):
        #Wavelet - !!!!!!
        img_filtered = cv2.bilateralFilter(img_Eq,9,75,75)
    if(i == 1):
        #TODO Bilateral
        img_filtered = cv2.bilateralFilter(img_Eq,9,75,75) # I can also apply different filters
    if(i == 2):
        #Anisotropic
        img_filtered = ad(img_Eq, niter=30,step= (19.,19.), kappa=50,gamma=0.25, option=1)
    if(i == 3):
        #GUIDED
        img_filtered = cv2.bilateralFilter(img_Eq,9,75,75)
    
    # change also parameters
    img_fr_1 = frangi(img_filtered,scale_range=3,scale_step=0.5,beta1=0.5,beta2= 0.01,  black_ridges=True)
    img_fr_2 = frangi(img_filtered,scale_range=4,scale_step=0.5,beta1=0.5,beta2= 0.01,  black_ridges=True)
    
    min_img = np.amin(img_fr_1,img_fr_2)
    
"""
kernel_fr = np.ones((5,5),np.uint8)
#img_fr = cv2.erode(img_fr,kernel_fr,iterations =1)
img_fr = cv2.morphologyEx(img_fr, cv2.MORPH_OPEN, kernel_fr)

MORPH = True
if MORPH:

    img_fr = img_as_ubyte(img_fr)
    #img_fr.astype(np.float32)
    _, img_thresh = cv2.threshold(img_fr,50,255,cv2.THRESH_BINARY)
    
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
    #_, defectImage = cv2.threshold(defectImage,200,255,cv2.THRESH_BINARY) # for now I couldn't realize any help of these
    
#TODO Hough
else:
    print("hough")

print(classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1]))))
"""
"""
new_img = np.zeros((100,100)) 

for i in range (10,50):
    for j in range(10,50):
        if i==j:
            new_img[i,j] = 1 
for i in range (10,50):
    for j in range(51,91):
        if j-i==41 :
            new_img[i,j] = 1 
img_fr2 = calculateVesselness2D(new_img, 1, 0.5, 0.05)
img_fr = frangi(new_img,scale_range=(0.5,1),scale_step=0.5,beta1=0.5,beta2= 0.05)
"""
"""
plt.subplot(2,2,1)
plt.imshow(img_fr,"gray"),plt.title('frangi25_3')
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

"""