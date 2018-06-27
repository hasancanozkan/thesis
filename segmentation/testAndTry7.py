'''
Created on 22.03.2018

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

from __builtin__ import str
import glob
from skimage.io import imsave



def callAsymmetricFrangi(img, sigma, beta, degreeOfAsymmetrie):
    if(degreeOfAsymmetrie == 1.0):
        return frangi(img,sigma_x = sigma,sigma_y = sigma,beta1=beta,  black_ridges=True)
    else:
        return frangi(img,sigma_x = sigma,sigma_y = sigma*degreeOfAsymmetrie,beta1=beta,  black_ridges=True), frangi(img,sigma_x = sigma*degreeOfAsymmetrie,sigma_y = sigma,beta1=beta,  black_ridges=True)

#mask of fourier
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006214_bad_.tif',0)
img_label = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000006214_bad_.tif',0)


image_list=[img_raw]
label_list=[img_label]

"""
for filename in glob.glob('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/*.tif'):
    img= cv2.imread(filename,0)
    label_list.append(img)
    
for filename in glob.glob('C:/Users/oezkan/HasanCan/RawImages/*.tif'):
    img= cv2.imread(filename,0)
    image_list.append(img)
"""




#start_time1 = time.time()
    
for im in range(len(image_list)):   
    # labeled signal
    _,labeled_crack = cv2.threshold(label_list[im],245,255,cv2.THRESH_BINARY)
    kernel_label = np.ones((1,1),np.uint8)   
    labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)
    
    # apply fft for grid fingers
    img_fft = fft(image_list[im], mask_fft) 
    
    # apply histogram equalization
    img_Eq = cv2.equalizeHist(img_fft)
    
    img_roi=createROI(img_Eq)
    
   
    #sigma x-y8
    param_x = [1.0]
    degree = np.arange(1,10.1,1)
    thresh = 0.20 

    
    for x_i in range(len(param_x)):
        #v=[]
        for d_i in range(len(degree)):
        
            
                                   
            if(degree[d_i] == 1.0):
                start_time1 = time.time()
                img_fr=frangi(img_fft,sigma_x = param_x[x_i],sigma_y = param_x[x_i],beta1=0.5,  black_ridges=True)
                print(time.time() - start_time1)
                """
                for t in range(len(img_fr)):
                    for z in range(len(img_fr)):
                    
                        if((img_fr[t][z])>thresh):
                        
                            img_fr[t][z] = 1
                            
                        else:
                            img_fr[t][z] = 0 
                        
                img_fr_max_roi = (img_fr*img_roi).astype(np.float32)
                imsave(str(im)+'AFmax_sigma_'+str(param_x[x_i])+'_degree'+str(degree[d_i])+'_thresh_0.20'+'.tif', img_fr_max_roi)"""
            else:
                start_time2 = time.time()
                img_fry = frangi(img_fft,sigma_x = param_x[x_i],sigma_y = param_x[x_i]*degree[d_i],beta1=0.5,  black_ridges=True) 
                print(time.time() - start_time2)
                start_time3 = time.time()
                img_frx = frangi(img_fft,sigma_x = param_x[x_i]*degree[d_i],sigma_y = param_x[x_i],beta1=0.5,  black_ridges=True)
                print(time.time() - start_time3)
                """
                for t in range(len(img_frx)):
                    for z in range(len(img_frx)):
                    
                        if((img_frx[t][z])>thresh):
                        
                            img_frx[t][z] = 1
                            
                        else:
                            img_frx[t][z] = 0 
                
                for t in range(len(img_fry)):
                    for z in range(len(img_fry)):
                    
                        if((img_fry[t][z])>thresh):
                        
                            img_fry[t][z] = 1
                            
                        else:
                            img_fry[t][z] = 0 
                                   
                img_fr_x_roi = (img_frx*img_roi).astype(np.float32)
                img_fr_y_roi = (img_fry*img_roi).astype(np.float32)
                imsave(str(im)+'AFmax_sigma_'+str(param_x[x_i])+'_x_degree'+str(degree[d_i])+'_thresh_0.20'+'.tif', img_fr_x_roi)
                imsave(str(im)+'AFmax_sigma_'+str(param_x[x_i])+'_y_degree'+str(degree[d_i])+'_thresh_0.20'+'.tif', img_fr_y_roi)"""