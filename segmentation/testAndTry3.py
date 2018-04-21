'''
Created on 22.03.2018

@author: oezkan
'''
from adaptedFrangiSelfBeta2 import frangi 
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
from skimage.io import imsave
from skimage.util.dtype import img_as_ubyte

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
    kernel_label = np.ones((1,1),np.uint8)   
    labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)
    start_time2 = time.time()
    # apply fft for grid fingers
    img_fft = fft(image_list[im], mask_fft) 
    
    # apply histogram equalization
    img_Eq = cv2.equalizeHist(img_fft)
    
    img_roi=createROI(img_Eq)
    print 'time for pre setting :'
    print(time.time() - start_time2)  
    #sigma x-y
    param_x = [0.75,1.0,1.25]
    #param_y = [0.5,0.75,1.0,1.25,1.5,2.0]
    #setting constants
    #param_beta1= [0.5]
    thresh = [0.2,0.25] # this means that we want to select X percent from probability map!!!!!
    
    class_result=""
    #apply frangi for different scales
    start_time1 = time.time()
    v=[]
    for i in range(len(thresh)):
        for k in range(len(param_x)):
                
            img_fr = frangi(img_fft,sigma_x = param_x[k],sigma_y = param_x[k],beta1=0.5,  black_ridges=True)          
                                   
       
            for t in range(len(img_fr)):
                for z in range(len(img_fr)):
                    
                    if((img_fr[t][z])>thresh[i]):
                        
                        img_fr[t][z] = 1
                            
                    else:
                        img_fr[t][z] = 0
                    
            v.append(img_fr)
    img_fr_min = v[0]                    
    for sigma_index in range(len(v)-1):
        img_fr_min = np.minimum(img_fr_min,v[sigma_index+1])
    print 'time for frangi'
    print(time.time() - start_time1)                
    img_fr_roi = (img_fr_min*img_roi).astype(np.float32)
     
    imsave(str(im)+'_AFvector_x_y.tif', img_fr_roi)                    
    

    # performance measure
    perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi) 
    class_result = class_result +str(perform_result)+"\n"
        
    #save the data
    with open(str(im)+'_AFvector_classification_results.txt','w') as output:
        output.write(str(class_result))
        
       