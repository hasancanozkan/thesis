'''
Created on 04.04.2018

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



#mask of fourier
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000007501_bad_.tif',0)
img_label = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000007501_bad_.tif',0)


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

  
    img_resize = cv2.resize(image_list[im],(512,512))
            
    img_down=cv2.pyrDown(image_list[im],(512,512))
    
    #Wavelet
    #img_fft /=255
    level = 2
    wavelet = 'haar'
    #decompose to 2nd level coefficients
    [cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)]  =  pywt.wavedec2(image_list[im], wavelet=wavelet,level=level)
    coeffs = [cA2,(cH2, cV2, cD2)]
    
    #reconstruction
    recon_img= pywt.waverec2(coeffs, wavelet=wavelet)
   
    # normalization to convert uint8
    img_filtered = cv2.normalize(recon_img,0 , 255, cv2.NORM_MINMAX)
    img_filtered *=255
    cv2.imwrite(str(im)+'_resized.tif', img_resize)
    #cv2.imwrite(str(im)+'_downScaled.tif', img_down)
    #cv2.imwrite(str(im)+'_Wavelet.tif', img_filtered)
    
    """
    plt.subplot(1,3,1),plt.imshow(img_down,'gray')
    plt.subplot(1,3,2),plt.imshow(img_resize,'gray')
    plt.subplot(1,3,3),plt.imshow(img_filtered,'gray')

    plt.show()
    """
            
                     