'''
Created on 01.04.2018

@author: oezkan
'''
from adaptedFrangi import frangi 
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
import multiprocessing as mp


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

def filtering(filterType, img_Eq, param1, param2):
        if filterType == 'bilateral':
            img_filtered= cv2.bilateralFilter(img_Eq,param1,param2,param2) 
            return img_filtered

def runFrangi(filtered_images,sigma_x,sigma_y,beta1,beta2,black_ridges=True):
    img_fr = img_as_ubyte(frangi(filtered_images,sigma_x ,sigma_y ,beta1,beta2,  black_ridges))
    return img_fr
    
for im in range(len(image_list)):      
    # apply fft for grid fingers
    img_fft = fft(image_list[im], mask_fft) 
    
    # apply histogram equalization
    img_Eq = cv2.equalizeHist(img_fft)
    
    img_roi=createROI(img_Eq)
    
    #list of parameters for filters
    param_bl = [[9,11],[100,150]]
    sigmas=[1.0,1.5,2.0]

    filtered_images = []
    img_fr =[]
    start_time1 = time.time()
    if __name__ == '__main__': 
        pool = mp.Pool(processes=2)
        pool2 = mp.Pool(4)
        filtered_images = [pool.apply_async(filtering, args = ('bilateral', img_Eq, param1, param2)) for param1 in param_bl[0] for param2 in param_bl[1]]
        filtered_images = [p.get() for p in filtered_images]
        
        img_fr = [pool2.apply_async(runFrangi, args =(filtered_image,x,y,0.5,0.125,True)) for x in sigmas 
                  for y in sigmas for filtered_image in filtered_images  ]
        img_fr = [p.get() for p in img_fr]
        
        #print len(img_fr)
        #plt.subplot(1,2,1),plt.imshow(img_fr[0],'gray')
        #plt.subplot(1,2,2),plt.imshow(img_fr[33],'gray')
        #plt.show()
print (time.time() - start_time1)   