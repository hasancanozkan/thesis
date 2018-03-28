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

            
            # performance measure
            perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi) 
            
            cv2.imwrite(str(im)+'_AF_bl_.tif', img_fr_roi)
            #save the data
            with open(str(im)+'_AF_bl_classification_results.txt','w') as output:
                output.write(str(perform_result))

        if(index == 1):
            #Anisotropic
            v = []
            for i in range(len(param_ad[0])):
                for j in range(len(param_ad[1])):
                    #Anisotropic Diffusion
                    img_filtered = ad(img_Eq, niter=param_ad[0][i],step= param_ad[1][j], kappa=50,gamma=0.10, option=1)
                    img_filtered=np.uint8(img_filtered)# if not frangi does not accept img_filtered because it is float between -1 and 1    
                    
    
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
        
            # performance measure
            perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi)
            
            cv2.imwrite(str(im)+'_AF_ad_.tif', img_fr_roi)
            
            #save the data
            with open(str(im)+'_AF_ad_classification_results.txt','w') as output:
                output.write(str(perform_result))
                        
        if(index == 2):
            v = []
            #GUIDED
            for i in range(len(param_gf[0])):
                for j in range(len(param_gf[1])):
                    #Guided
                    img_filtered = guidedFilter(img_Eq, img_Eq, param_gf[0][i], param_gf[1][j]) 
                                  
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

            
            # performance measure
            perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi)
            
            cv2.imwrite(str(im)+'_AF_gf_.tif', img_fr_roi)

            #save the data
            with open(str(im)+'_AF_gf_classification_results.txt','w') as output:
                output.write(str(perform_result))
        
        if (index == 3):
            
            img_roi=cv2.pyrDown(img_roi,(512,512))
            #Wavelet
            #img_fft /=255
            level = 2
            wavelet = 'haar'
            #decompose to 2nd level coefficients
            [cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)]  =  pywt.wavedec2(img_Eq, wavelet=wavelet,level=level)
            coeffs = [cA2,(cH2, cV2, cD2)]
            
            #reconstruction
            recon_img= pywt.waverec2(coeffs, wavelet=wavelet)
           
            # normalization to convert uint8
            img_filtered = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
            #img_filtered = recon_img*255 # !!!!!!!!!!!bu hatali 
            v = [] 
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

            #to have classification result downscale
            labeled_crack = cv2.pyrDown(labeled_crack,(512,512))        
            _,labeled_crack = cv2.threshold(labeled_crack,3,255,cv2.THRESH_BINARY)     

            # performance measure
            perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi)
            
            cv2.imwrite(str(im)+'_AF_waveApprox.tif', img_fr_roi)
            
            #save the data
            with open(str(im)+'_AF_waveApprox_classification_results.txt','w') as output:
                output.write(str(perform_result))


print(time.time() - start_time1)
