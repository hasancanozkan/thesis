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
    
   
    #sigma x-y8
    param_x = [0.75,1.0,1.25]
    #param_y = np.arange(0.5,4.76,0.25)
    thresh = [0.20]
    #list of parameters for filters
    param_bl0 = np.arange(5,8,2)
    param_bl1= [100]
    param_ad0 = [3]
    param_ad1= [1.0]
    
    bl_class_result = ""
    bl_class_result_max=""
    ad_class_result = ""
    ad_class_result_max = ""
    
    
    for index in range (0,2,1): 
        if(index == 0):
            for l in range(len(param_bl1)):
                for i in range (len(param_bl0)):
                    v= []
                    for t_i in range(len(thresh)):
                        for x_i in range(len(param_x)):
                        
                        
                            img_filtered = cv2.bilateralFilter(img_fft,param_bl0[i],param_bl1[l],param_bl1[l])
        
                            v.append(frangi(img_filtered,sigma_x = param_x[x_i],sigma_y = param_x[x_i],beta1=0.5,  black_ridges=True))
                        img_fr_max = img_fr_min = v[0]
                        
                        # this part can be optimized with where!! check the np maximum page
                        for sigma_index in range(len(v)-1):
                            img_fr_min = np.minimum(img_fr_min,v[sigma_index+1])   
                        for sigma_index in range(len(v)-1):
                            img_fr_max = np.maximum(img_fr_max,v[sigma_index+1])   
                            
                        # THE part below can be optimized??????????
                        for t in range(len(img_fr_min)):
                            for z in range(len(img_fr_min)):
                                
                                if((img_fr_min[t][z])>thresh[t_i]):
                                    
                                    img_fr_min[t][z] = 1
                                        
                                else:
                                    img_fr_min[t][z] = 0   
                                    
                        for t in range(len(img_fr_max)):
                            for z in range(len(img_fr_max)):
                                
                                if((img_fr_max[t][z])>thresh[t_i]):
                                    
                                    img_fr_max[t][z] = 1
                                        
                                else:
                                    img_fr_max[t][z] = 0 
                                    
                        img_fr_min_roi = (img_fr_min*img_roi).astype(np.float32)
                        img_fr_max_roi = (img_fr_max*img_roi).astype(np.float32)
                        
                        imsave(str(im)+'AFmin_thresh_'+str(thresh[t_i])+'_bl0_'+str(param_bl0[i])+'_bl1_'+str(param_bl1[l])+'.tif', img_fr_min_roi)
                        imsave(str(im)+'AFmax_thresh_'+str(thresh[t_i])+'_bl0_'+str(param_bl0[i])+'_bl1_'+str(param_bl1[l])+'.tif', img_fr_max_roi)                                                   
                        # performance measure
                        
                        perform_result = sensitivity, _, _, _, onlyFP=perf_measure(labeled_crack, img_fr_min_roi, img_roi)
                        perform_result_max = sensitivity_max, _, _, _, onlyFP_max=perf_measure(labeled_crack, img_fr_max_roi, img_roi)
                        
                        bl_class_result = bl_class_result+'_thresh_'+str(thresh[t_i])+"_bl0_"+str(param_bl0[i])+"\n" + str(perform_result)+"\n"
                        bl_class_result_max = bl_class_result_max+'_thresh_'+str(thresh[t_i])+"_bl0_"+str(param_bl0[i])+"\n" + str(perform_result_max)+"\n"
                        #save the data
                        with open(str(im)+'_AFmin_bl_classification_results'+'_thresh_'+str(thresh[t_i])+'.txt','w') as output:
                            output.write(str(bl_class_result))
                        #save the data
                        with open(str(im)+'_AFmax_bl_classification_results'+'_thresh_'+str(thresh[t_i])+'.txt','w') as output:
                            output.write(str(bl_class_result_max))    
                
        if(index == 1):
            for l in range(len(param_ad1)):                        
                for i in range (len(param_ad0)):
                    v= []
                    for t_i in range(len(thresh)):
                        for x_i in range(len(param_x)):
                            
                            img_filtered = ad(img_fft, niter=param_ad0[i],step= (param_ad1[l],param_ad1[l]), kappa=50,gamma=0.10, option=1)
                            img_filtered=np.uint8(img_filtered)# if not frangi does not accept img_filtered because it is float between -1 and 1
        
                            v.append(frangi(img_filtered,sigma_x = param_x[x_i],sigma_y = param_x[x_i],beta1=0.5,  black_ridges=True))
                                                       
                        img_fr_max = img_fr_min = v[0]
                         
                        for sigma_index in range(len(v)-1):
                            img_fr_min = np.minimum(img_fr_min,v[sigma_index+1])   
                        for sigma_index in range(len(v)-1):
                            img_fr_max = np.maximum(img_fr_max,v[sigma_index+1])   
                            
                        # THE part below can be optimized??????????
                        for t in range(len(img_fr_min)):
                            for z in range(len(img_fr_min)):
                                
                                if((img_fr_min[t][z])>thresh[t_i]):
                                    
                                    img_fr_min[t][z] = 1
                                        
                                else:
                                    img_fr_min[t][z] = 0   
                                    
                        for t in range(len(img_fr_max)):
                            for z in range(len(img_fr_max)):
                                
                                if((img_fr_max[t][z])>thresh[t_i]):
                                    
                                    img_fr_max[t][z] = 1
                                        
                                else:
                                    img_fr_max[t][z] = 0 
                                    
                        img_fr_min_roi = (img_fr_min*img_roi).astype(np.float32)
                        img_fr_max_roi = (img_fr_max*img_roi).astype(np.float32)
                        imsave(str(im)+'AFmin_thresh_'+str(thresh[t_i])+'_ad0_'+str(param_ad0[i])+'.tif', img_fr_min_roi)  
                        imsave(str(im)+'AFmax_thresh_'+str(thresh[t_i])+'_ad0_'+str(param_ad0[i])+'.tif', img_fr_max_roi)                                                 
                        # performance measure
                        perform_result = sensitivity, _, _, _, onlyFP=perf_measure(labeled_crack, img_fr_min_roi, img_roi)
                        perform_result = sensitivity, _, _, _, onlyFP=perf_measure(labeled_crack, img_fr_max_roi, img_roi)
                        
                        ad_class_result = ad_class_result+'_thresh_'+str(thresh[t_i])+"_ad0_"+str(param_ad0[i])+"\n" + str(perform_result)+"\n"
                        ad_class_result_max = ad_class_result_max+'_thresh_'+str(thresh[t_i])+"_ad0_"+str(param_ad0[i])+"\n" + str(perform_result_max)+"\n"
                
                        #save the data
                        with open(str(im)+'_AFmin_ad_classification_results'+'_thresh_'+str(thresh[t_i])+'.txt','w') as output:
                            output.write(str(ad_class_result))
                        #save the data
                        with open(str(im)+'_AFmax_ad_classification_results'+'_thresh_'+str(thresh[t_i])+'.txt','w') as output:
                            output.write(str(ad_class_result_max))