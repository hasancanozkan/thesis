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
import math


img_raw =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000693108.tif',0)
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
labeled_crack = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736_CrackLabel.tif') # as a new ROI
#labeled_crack2 = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000006604_bad_.tif',0)
#img_raw2 = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006604_bad_.tif',0)
maskRoi = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736_BusLabel.tif',0) 



_, labeled_crack = cv2.threshold(labeled_crack[:,:,2],127,255,cv2.THRESH_BINARY)
kernel_label = np.ones((2,2),np.uint8)
labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)

# apply fft for grid fingers
img_fft = fft(maskRoi, mask_fft)

# apply histogram equalization
img_Eq = cv2.equalizeHist(img_fft)

#list of parameters for filters
param_bl = [[5,9],[25,50]]
param_gf = [[2,3,4,5,6,7],[0.1,0.2,0.3,0.4]]
param_ad = [[3,5,10,15,20],[(0.5,0.5),(1,1),(2,2),(5,5),(10,10)]]

param_scale = [(0.5,0.6),(1.0,1.1),(1.5,1.6),(2.0,2.1),(2.5,2.6),(3.0,3.6),(3.5,3.6),(4.0,4.1)]

wave_class_result = ""
#apply the filters

#img_filtered2 = ad(img_Eq, niter=param_ad[0][0],step=param_ad[1][0], kappa=50,gamma=0.10, option=1)

#plt.imshow(img_filtered2,'gray')
#plt.show()

for i in range (0,4,1):
    if(i==0):
        #Wavelet
        def mad(arr):
            """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample. 
        """
            arr = np.ma.array(arr).compressed()
            med = np.median(arr)
            return np.median(np.abs(arr - med))
        img_Eq /=255
        level = 2
        wavelet = 'haar'
        #decompose to 2nd level coefficients
        coeffs=  pywt.wavedec2(img_Eq, wavelet=wavelet,level=level)
    

        #calculate the threshold
        sigma = mad(coeffs[-level])
        threshold = sigma*np.sqrt( 2*np.log(img_Eq.size/2)) #this is soft thresholding
        #threshold = 50
        newCoeffs = map (lambda x: pywt.threshold(x,threshold,mode='hard'),coeffs)

        #reconstruction
        recon_img= pywt.waverec2(coeffs, wavelet=wavelet)
        v = [[],[],[],[],[],[],[],[]]
        # normalization to convert uint8
        img_filtered = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
        img_filtered *=255
        #apply frangi for different scales
        for k in range(len(param_scale)):
            v[k] = frangi(img_filtered,scale_range=param_scale[k],scale_step=1,beta1=0.5,beta2= 0.05,  black_ridges=True)
            v[k] = img_as_ubyte(v[k])

        min_img = np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])
                    
        #check f! score of all possibilities
        new_result = (classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),min_img.reshape((min_img.shape[0]*min_img.shape[1]))))
        wave_class_result = wave_class_result+new_result
        
        #save the data
        with open('wave_classification_results.txt','w') as output:
            output.write(wave_class_result)
    if(i == 1):
        #Anisotropic
        img_filtered = ad(img_Eq, niter=30,step= (19.,19.), kappa=50,gamma=0.25, option=1)
    if(i == 2):
        #GUIDED
        img_filtered = cv2.bilateralFilter(img_Eq,9,75,75)
    if (i == 3):
        #Wavelet - !!!!!!
        img_filtered = cv2.bilateralFilter(img_Eq,9,75,75) 




#with open('out2.txt','w') as output:
 #   output.write(classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),img_fr2.reshape((img_fr2.shape[0]*img_fr2.shape[1]))))
"""
#list of parameters for filters
param_bl = [[5,9],[75,100]]
param_gf = [[2,3,4,5,6,7],[0.1,0.2,0.3,0.4]]
param_ad = [[3,5,10,15,20],[0.5,1,2,5,10]]

param_scale = [(0.5,0.6),(1.0,1.1)]



#apply the filters



img_frangi = [[],[]]
for i in range(len(param_bl[0])):
    for j in range(len(param_bl[1])):
        #Bilateral
        img_filtered = cv2.bilateralFilter(img_Eq,param_bl[0][i],param_bl[1][j],param_bl[1][j])
                
        #apply frangi for different scales
        for k in range(len(param_scale)):
            img_frangi[k] = frangi(img_filtered,scale_range=param_scale[k],scale_step=1,beta1=0.5,beta2= 0.05,  black_ridges=True)
                    
        min_img = np.minimum(img_frangi[0],img_frangi[1])
                    
        #check f! score of all possibilities
        print(classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),min_img.reshape((min_img.shape[0]*min_img.shape[1]))))

        #save the data
        with open('out.txt','w') as output:
            output.write(classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),min_img.reshape((min_img.shape[0]*min_img.shape[1]))))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111
#19.02  after meeting Daniel
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
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
"""
"""
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
    img_fr1 = frangi(img_filtered,scale_range=(0.1,0.2),scale_step=1,beta1=0.5,beta2= 0.001,  black_ridges=True)*labeled_crack
    img_fr2 = frangi(img_filtered,scale_range=(1,1.1),scale_step=1,beta1=0.5,beta2= 0.001,  black_ridges=True)
    min_img = np.minimum(img_fr1,img_fr2)
   """ 
