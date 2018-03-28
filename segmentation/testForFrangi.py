'''
Created on 21.03.2018

@author: oezkan
'''
from skimage.filters import frangi 
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
from performanceMeasure import perf_measure
from newROI import createROI
from fftFunction import fft
from AnisotropicDiffusionSourceCode import anisodiff as ad
from cv2.ximgproc import guidedFilter
import pywt
import glob

#mask of fourier
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000007373_bad_.tif',0)
img_label = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000007373_bad_.tif',0)


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
    param_bl = [[9,11],[100,150]]#changed
    param_gf = [[2,4],[0.2,0.4]]
    param_ad = [[10,15],[(0.5,0.5),(1.,1.)]]#changed
    

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
                       
                        v.append(img_as_ubyte(frangi(img_filtered, (1.0,2.1), 0.5, 0.5, 0.125, black_ridges=True))) 
                        
                        
                img_fr = v[0]  
                for sigma_index in range(len(v)-1):
                    img_fr = np.maximum(img_fr,v[sigma_index+1])                        
                _, img_thresh = cv2.threshold(img_fr,0,255,cv2.THRESH_BINARY) 
                            
                img_fr_roi = img_thresh*img_roi                                                

                # performance measure
                perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi)
                    
                cv2.imwrite(str(im)+'_Frangi_bl_.tif', img_fr_roi)
           
                #save the data
                with open(str(im)+'_Frangi_bl_classification_results.txt','w') as output:
                    output.write(str(perform_result))
    
            if(index == 1):
                v=[]
                #Anisotropic                
                for i in range(len(param_ad[0])):
                    for j in range(len(param_ad[1])):
                        #Anisotropic Diffusion
                        img_filtered = ad(img_Eq, niter=param_ad[0][i],step= param_ad[1][j], kappa=50,gamma=0.10, option=1)
                        img_filtered=np.uint8(img_filtered)# if not frangi does not accept img_filtered because it is float between -1 and 1    
                        
                        v.append(img_as_ubyte(frangi(img_filtered, (1.0,2.1), 0.5, 0.5, 0.125, black_ridges=True))) 
                                                
                img_fr = v[0]  
                for sigma_index in range(len(v)-1):
                    img_fr = np.maximum(img_fr,v[sigma_index+1])                        
                _, img_thresh = cv2.threshold(img_fr,0,255,cv2.THRESH_BINARY) 
                                                    
                img_fr_roi = img_thresh*img_roi                                                

                # performance measure
                perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi)
                    
                cv2.imwrite(str(im)+'_Frangi_ad_.tif', img_fr_roi)
           
                #save the data
                with open(str(im)+'_Frangi_ad_classification_results.txt','w') as output:
                    output.write(str(perform_result))
            
            if(index == 2):
                #GUIDED
                for i in range(len(param_gf[0])):
                    for j in range(len(param_gf[1])):
                        #Guided
                        img_filtered = guidedFilter(img_Eq, img_Eq, param_gf[0][i], param_gf[1][j]) 
   
                        v.append(img_as_ubyte(frangi(img_filtered, (1.0,2.1), 0.5, 0.5, 0.125, black_ridges=True))) 
                                              
                img_fr = v[0]  
                for sigma_index in range(len(v)-1):
                    img_fr = np.maximum(img_fr,v[sigma_index+1])                        
                _, img_thresh = cv2.threshold(img_fr,0,255,cv2.THRESH_BINARY) 
                                     
                img_fr_roi = img_thresh*img_roi                                                
               
                # performance measure
                perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi)
                    
                cv2.imwrite(str(im)+'_Frangi_gf_.tif', img_fr_roi)
           
                #save the data
                with open(str(im)+'_Frangi_gf_classification_results.txt','w') as output:
                    output.write(str(perform_result))
                    
            if (index == 3):
                
                #Wavelet
                img_roi=cv2.pyrDown(img_roi,(512,512))
                #img_fft /=255
                level = 2
                wavelet = 'haar'
                #decompose to 2nd level coefficients
                [cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)]  =  pywt.wavedec2(img_Eq, wavelet=wavelet,level=level)
                coeffs = [cA2,(cH2, cV2, cD2)]
                #calculate the threshold
                #sigma = mad(coeffs[-level])
                #threshold = sigma*np.sqrt( 2*np.log(img_Eq.size/2)) #this is soft thresholding
                #print threshold
                #threshold = 30
                #newCoeffs = map (lambda x: pywt.threshold(x,threshold*2,mode='hard'),coeffs)
                
                #reconstruction
                recon_img= pywt.waverec2(coeffs, wavelet=wavelet)
               
                # normalization to convert uint8
                img_filtered = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
                #plt.imshow(img_filtered,'gray'),plt.show()
                 
                img_fr = img_as_ubyte(frangi(img_filtered, (1.0,2.1), 0.5, 0.5, 0.125, black_ridges=True))
                _, img_thresh = cv2.threshold(img_fr,0,255,cv2.THRESH_BINARY)
                
                img_fr_roi = img_thresh*img_roi                                                
           
                
                labeled_crack=cv2.pyrDown(labeled_crack,(512,512))
                _, labeled_crack = cv2.threshold(labeled_crack,5,255,cv2.THRESH_BINARY)
                
                # performance measure
                perform_result = perf_measure(labeled_crack, img_fr_roi, img_roi)
                    
                cv2.imwrite(str(im)+'_Frangi_waweApprox_.tif', img_fr_roi)
           
                #save the data
                with open(str(im)+'_Frangi_waweApprox_classification_results.txt','w') as output:
                    output.write(str(perform_result))
            
print(time.time() - start_time1)





