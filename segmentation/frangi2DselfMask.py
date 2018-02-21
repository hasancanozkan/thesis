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
from AnisotropicDiffusionSourceCode import anisodiff as ad
from cv2.ximgproc import guidedFilter


img_raw =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736.tif',0)
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
labeled_crack = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736_CrackLabel.tif')
maskRoi = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736_BusLabel.tif',0) 

# labeled signal 
_, labeled_crack = cv2.threshold(labeled_crack[:,:,2],127,255,cv2.THRESH_BINARY)
kernel_label = np.ones((2,2),np.uint8)
labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)

#apply mask for busbars and ROI, for now artificially drawn
# !!! it can not work as I do now !!!! I have to change convolve it with mask


# apply fft for grid fingers
img_fft = fft(maskRoi, mask_fft)

# apply histogram equalization
img_Eq = cv2.equalizeHist(img_fft)

#list of parameters for filters
param_bl = [[5,9,11,13,15],[25,50,75,100,125,150]]
param_gf = [[2,3,4,5,6,7],[0.1,0.2,0.3,0.4]]
param_ad = [[3,5,10,15,20],[0.5,1,2,5,10]]

param_scale = [(0.5,0.6),(1.0,1.1),(1.5,1.6),(2.0,2.1),(2.5,2.6),(3.0,3.6),(3.5,3.6),(4.0,4.1)]
print param_scale[0]
bl_class_result = ""

#apply the filters
for i in range (0,4,1):
    
    if(i == 0):
        v = [[],[],[],[],[],[],[],[]]
        for i in range(len(param_bl[0])):
            for j in range(len(param_bl[1])):
                #Bilateral
                img_filtered = cv2.bilateralFilter(img_Eq,param_bl[0][i],param_bl[1][j],param_bl[1][j])
                
                #apply frangi for different scales
                for k in range(len(param_scale)):
                    v[k] = frangi(img_filtered,scale_range=param_scale[k],scale_step=1,beta1=0.5,beta2= 0.05,  black_ridges=True)
                    v[k] = img_as_ubyte(v[k])

                min_img = np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])
                    
                #check f! score of all possibilities
                new_result = (classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),min_img.reshape((min_img.shape[0]*min_img.shape[1]))))
                bl_class_result = bl_class_result+"bl_i_"+str(i)+"_j_"+str(j)+new_result
                #save the data
                with open('bl_classification_results.txt','w') as output:
                    output.write(bl_class_result)
    if(i == 1):
        #Anisotropic
        img_filtered = ad(img_Eq, niter=30,step= (19.,19.), kappa=50,gamma=0.25, option=1)
    if(i == 2):
        #GUIDED
        img_filtered = cv2.bilateralFilter(img_Eq,9,75,75)
    if (i == 3):
        #Wavelet - !!!!!!
        img_filtered = cv2.bilateralFilter(img_Eq,9,75,75)   





""" # for now I wont use this part of code
plt.imshow(img_fr,"gray"),plt.title('img_fr')
plt.xticks([]), plt.yticks([])
plt.show()

kernel_fr = np.ones((5,5),np.uint8)
#img_fr = cv2.erode(img_fr,kernel_fr,iterations =1)
img_fr = cv2.morphologyEx(img_fr, cv2.MORPH_OPEN, kernel_fr)

MORPH = True
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

#print(classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1]))))

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