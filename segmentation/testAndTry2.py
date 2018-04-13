'''
Created on 12.04.2018

@author: oezkan
'''
from adaptedFrangiSelfBeta2 import frangi 
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
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


#start_time1 = time.time()
    
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

    
    #sigma x-y
    param_x = [0.75,1.0,1.25,1.5]
    param_y = [0.75,1.0,1.25,1.5]
    thresh = [0]
    param_beta1= [0.5]
    class_result = ''
    #apply the filters
    for index in range (0,4,1):
    
        if(index == 0):
            start_time1 = time.time()
            for i in range(len(param_beta1)):
                for j in range(len(thresh)):
                    #Bilateral
                    
                    #apply frangi for different scales
                    for k in range(len(param_x)):
                        for l in range(len(param_y)):
                            img_fr = img_as_ubyte(frangi(img_Eq,sigma_x = param_x[k],sigma_y = param_y[l],beta1=param_beta1[i], black_ridges=True))
                                 
                            _, img_thresh = cv2.threshold(img_fr,thresh[j],255,cv2.THRESH_BINARY)
                            img_fr_roi = img_thresh*img_roi                                                
                
                            # performance measure
                            perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi)
                            
                            class_result = class_result+"_x_"+str(param_x[k])+"_y_"+str(param_y[l])+'_beta1_'+str(param_beta1[i])+'_thresh_'+str(thresh[j])+"\n" + str(perform_result)+"\n"
                            cv2.imwrite(str(im)+'_AFself_'+"_x_"+str(param_x[k])+"_y_"+str(param_y[l])+'_beta1_'+str(param_beta1[i])+'_thresh_'+str(thresh[j])+'.tif', img_fr_roi)
                            #save the data
                            with open(str(im)+'_AFself_classification_results.txt','w') as output:
                                output.write(str(class_result))
            print(time.time() - start_time1) 

            
            