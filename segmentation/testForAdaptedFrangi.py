'''
Created on 26.02.2018

@author: oezkan
'''
from adaptedFrangi import frangi 
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report
from newROI import createROI, createROIfromOriginal
from fftFunction import fft
from AnisotropicDiffusionSourceCode import anisodiff as ad
from cv2.ximgproc import guidedFilter
import pywt
from __builtin__ import str
import glob

'''
Created on 26.02.2018

@author: oezkan
'''


#mask of fourier
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000007373_bad_.tif',0)
img_label = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000007373_bad_.tif',0)


image_list=[img_label]
label_list=[img_raw]
"""
for filename in glob.glob('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/*.tif'):
    img= cv2.imread(filename,0)
    label_list.append(img)
    
for filename in glob.glob('C:/Users/oezkan/HasanCan/RawImages/*.tif'):
    img= cv2.imread(filename,0)
    image_list.append(img)
"""

start_time1 = time.time()
    
for im in range(len(image_list)):    
    # labeled signal
    
    '''since Andreas' labels area so thin I need to enlarge them
    '''
    _,labeled_crack = cv2.threshold(label_list[im],245,255,cv2.THRESH_BINARY)
    kernel_label = np.ones((3,3),np.uint8)   
    labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)
    
    #plt.imshow(labeled_crack,'gray'),plt.title('label')
    #plt.show()
    
    #apply mask for busbars and ROI, for now artificially drawn
    # !!! it can not work as I do now !!!! I have to change convolve it with mask
    
    
    '''be sure that image is either img_raw or white labelede maskRoi
    '''
    # apply fft for grid fingers
    img_fft = fft(image_list[im], mask_fft) 
    
    # apply histogram equalization
    img_Eq = cv2.equalizeHist(img_fft)
    
    '''this part is my own roi, which is bad with star cracks
    then multiply roi with tghe result of frangi
    '''
    img_roi=createROI(img_Eq)
    img_roi_fft = createROIfromOriginal(img_fft)
    img_roi = img_roi*img_roi_fft
    cv2.imwrite(str(im)+'.tif', img_roi*255)
    #list of parameters for filters
    param_bl = [[9,11],[100,150]]#changed
    param_gf = [[2],[0.2]]
    param_ad = [[10,15],[(0.5,0.5),(1.,1.)]]#changed
    
    #sigma x-y
    param_x = [5.]#changed
    param_y = [5.]#changed
    #setting constants
    param_beta1= [0.5]
    param_beta2 = [0.125]
    
    bl_class_result = ""
    ad_class_result = ""
    gf_class_result = ""
    wave_class_result = "" 
    
    #apply the filters
    for index in range (0,4,1):
                        
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
            #img_fr = img_fr*img_roi                                                
            ''' this part of the code applies old algorithm to get rid of unwanted shapes
            may be I can delete this part later
            '''
            
            #img_fr.astype(np.float32)
            _, img_thresh = cv2.threshold(img_fr,0,255,cv2.THRESH_BINARY)
            img_fr_roi = img_thresh*img_roi                                                
         
            
            #check f! score of all possibilities
            new_result = (classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),img_thresh.reshape((img_thresh.shape[0]*img_thresh.shape[1]))))
            gf_class_result = gf_class_result+"gf_"+str(param_gf[0][i])+"_"+str(param_gf[1][j])+ "_beta1_" + str(param_beta1[t]) + "_beta2_" + str(param_beta2[z]) +"\n" +new_result
            
            cv2.imwrite('Afrangi'+str(im)+'gf_'+str(param_gf[0][i])+'_'+str(param_gf[1][j])+ '_beta1_' + str(param_beta1[t]) + '_beta2_' + str(param_beta2[z])+'.tif', img_thresh)
            cv2.imwrite('Afrangi'+str(im)+'ROI_gf_'+str(param_gf[0][i])+'_'+str(param_gf[1][j])+ '_beta1_' + str(param_beta1[t]) + '_beta2_' + str(param_beta2[z])+'.tif', img_fr_roi)

            #save the data
            #with open(str(im)+'_gf_classification_results.txt','w') as output:
             #   output.write(gf_class_result)
        
        
print(time.time() - start_time1)





