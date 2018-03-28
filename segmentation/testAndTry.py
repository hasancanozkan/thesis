'''
Created on 22.03.2018

@author: oezkan
'''
from adaptedFrangi import frangi 
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report
from newROI import createROI
from fftFunction import fft
from AnisotropicDiffusionSourceCode import anisodiff as ad
from cv2.ximgproc import guidedFilter
import pywt
from __builtin__ import str
import glob
from performanceMeasure import perf_measure


#mask of fourier
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006214_bad_.tif',0)
img_label = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000006214_bad_.tif',0)


image_list=[]
label_list=[]

for filename in glob.glob('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/*.tif'):
    img= cv2.imread(filename,0)
    label_list.append(img)
    
for filename in glob.glob('C:/Users/oezkan/HasanCan/RawImages/*.tif'):
    img= cv2.imread(filename,0)
    image_list.append(img)


start_time1 = time.time()
    
for im in range(len(image_list)):    
    # labeled signal
    _,labeled_crack = cv2.threshold(label_list[im],245,255,cv2.THRESH_BINARY)
    kernel_label = np.ones((3,3),np.uint8)   
    labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)
    
    # apply fft for grid fingers
    img_fft = fft(image_list[im], mask_fft) 
    
    # apply histogram equalization
    img_Eq = cv2.equalizeHist(img_fft)
    
    img_roi=createROI(img_Eq)
    
    #list of parameters for filters
    param_bl = [[9,11],[100,150]]
    param_gf = [[2,4],[0.2,0.4]]
    param_ad = [[10,15],[(0.5,0.5),(1.,1.)]]
    
    #sigma x-y
    param_x = [1.0,1.5,2.0]
    param_y = [1.0,1.5,2.0]
    #setting constants
    param_beta1= [0.5]
    param_beta2 = [0.125]
    
    bl_class_result = ""
    ad_class_result = ""
    gf_class_result = ""
    wave_class_result = "" 
    
    #apply the filters
    for index in range (0,4,1):
        
        if(index == 0):
            v = []
            for i in range(len(param_bl[0])):
                for j in range(len(param_bl[1])):
                    #Bilateral
                    img_filtered = cv2.bilateralFilter(img_Eq,param_bl[0][i],param_bl[1][j],param_bl[1][j])
                   
                    #apply frangi for different scales
                    for t in range(len(param_beta1)):
                        for z in range(len(param_beta2)):
                            
                            for k in range(len(param_x)):
                                for l in range(len(param_y)):
                                        v.append(img_as_ubyte(frangi(img_filtered,sigma_x = param_x[k],sigma_y = param_y[l],beta1=param_beta1[t],beta2= param_beta2[z],  black_ridges=True))) 
                                         
                                

            img_fr = v[0]
            for sigma_index in range(len(v)-1):
                img_fr = np.minimum(img_fr,v[sigma_index+1])
            
            _, img_thresh = cv2.threshold(img_fr,0,255,cv2.THRESH_BINARY)
            img_fr_roi = img_thresh*img_roi                                                
            plt.subplot(1,3,1),plt.imshow(labeled_crack,'gray')
            plt.subplot(1,3,2),plt.imshow(img_fr_roi,'gray')
            plt.subplot(1,3,3),plt.imshow(img_roi,'gray')
            plt.show()
            
            # performance measure
            perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi) 
            
            #cv2.imwrite(str(im)+'bl_'+str(param_bl[0][i])+'_'+str(param_bl[1][j])+ '_beta1_' + str(param_beta1[t]) + '_beta2_' + str(param_beta2[z])+'.tif', img_fr_roi)
            #save the data
            with open(str(im)+'_bl_classification_results.txt','w') as output:
                output.write(str(perform_result))