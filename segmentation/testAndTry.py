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
from skimage.io import imsave



def callAsymmetricFrangi(img, sigma, beta, degreeOfAsymmetrie):
    if(degreeOfAsymmetrie == 1.0):
        return frangi(img,sigma_x = sigma,sigma_y = sigma,beta1=beta,  black_ridges=True)
    else:
        return np.maximum(frangi(img,sigma_x = sigma,sigma_y = sigma*degreeOfAsymmetrie,beta1=beta,  black_ridges=True), frangi(img,sigma_x = sigma*degreeOfAsymmetrie,sigma_y = sigma,beta1=beta,  black_ridges=True))

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
    if(im==15 or im==34 or im==13 or im==41):
        # labeled signal
        _,labeled_crack = cv2.threshold(label_list[im],245,255,cv2.THRESH_BINARY)
        kernel_label = np.ones((1,1),np.uint8)   
        labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)
        
        # apply fft for grid fingers
        img_fft = fft(image_list[im], mask_fft) 
        
        # apply histogram equalization
        img_Eq = cv2.equalizeHist(img_fft)
        
        img_roi=createROI(img_Eq)
        
        #img_roi = cv2.resize(img_roi,(512,512))

        #sigma x-y8
        param_x = [1.0]
        degree = np.arange(1.5,1.6,1)
        thresh = 0.20 
    
        
        for x_i in range(len(param_x)):
            #v=[]
            for d_i in range(len(degree)):
                img_filtered = cv2.bilateralFilter(img_fft,5,100,100)
                #img_filtered = ad(img_fft, niter=3, kappa=50,gamma=0.10, option=1)
                #img_filtered=np.uint8(img_filtered)# if not frangi does not accept img_filtered because it is float between -1 and 1
                #img_filtered = guidedFilter(img_fft, img_fft, 2, 0.2**2)
                #[cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)]  =  pywt.wavedec2(img_fft, wavelet='haar',level=2)
                #coeffs = [cA2,(cH2, cV2, cD2)]
                #reconstruction
                #recon_img= pywt.waverec2(coeffs, wavelet='haar')


                # normalization 
                #img_filtered = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
                
                img_fr_max = callAsymmetricFrangi(img_filtered, param_x[x_i], 0.5, degree[d_i])
                                       
                
            #for sigma_index in range(len(v)-1):
                #img_fr_max = np.maximum(img_fr_max,v[sigma_index+1])   
                
            # THE part below can be optimized?????????           
                for t in range(len(img_fr_max)):
                    for z in range(len(img_fr_max)):
                        
                        if((img_fr_max[t][z])>thresh):
                            
                            img_fr_max[t][z] = 1
                                
                        else:
                            img_fr_max[t][z] = 0 
                            
                img_fr_max_roi = (img_fr_max*img_roi).astype(np.float32)
                imsave(str(im)+'AFmax_bl_5_sigma_'+str(param_x[x_i])+'_degree'+str(degree[d_i])+'_thresh_0.20'+'.tif', img_fr_max_roi)       
            
