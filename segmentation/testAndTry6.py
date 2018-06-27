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
from AnisotropicDiffusionSourceCode import anisodiff as ad
from cv2.ximgproc import guidedFilter
import pywt
from __builtin__ import str
import glob
from performanceMeasure import perf_measure
from skimage.io import imsave
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import cStringIO
from PIL import Image


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
    
    # apply fft for grid fingers
    img_fft = fft(image_list[im], mask_fft) 
    
    # apply histogram equalization
    img_Eq = cv2.equalizeHist(img_fft)
    
    img_roi=createROI(img_Eq)
    
   
    #sigma x-y
    param_x = np.arange(0.5,2.01,0.25)
    param_y = np.arange(0.5,2.01,0.25)
    #setting constants
    
    thresh = np.arange(0.15,0.26,0.05)
    #thresh= [0.2,0.25]
    
    class_result = ""
    #apply the filters
    for i in range (len(param_x)):
        
        start_time1 = time.time()
        
        for l in range(len(thresh)):
            onlyFPvec = []
            sensitivityVec=[]
            for k in range(len(param_y)):
                
                img_fr = frangi(img_fft,sigma_x = param_x[i],sigma_y = param_y[k],beta1=0.5,  black_ridges=True)
                #save the frangi image in order to check it with IJ
                #imsave(str(im)+'_img_fr_bl_x_y_'+str(param_x[k])+'_thresh_'+str(thresh[l])+'.tif', (img_fr).astype(np.float32))
                
                # THE part below can be optimized??????????
                for t in range(len(img_fr)):
                    for z in range(len(img_fr)):
                        
                        if((img_fr[t][z])>thresh[l]):
                            
                            img_fr[t][z] = 1
                                
                        else:
                            img_fr[t][z] = 0    
                img_fr_roi = (img_fr*img_roi).astype(np.float32)
                imsave(str(im)+'_AF_x_'+str(param_x[k])+'_y_'+str(param_y[i])+'_thresh_'+str(thresh[l])+'.tif', img_fr_roi)                                                   
                

        print(time.time() - start_time1)