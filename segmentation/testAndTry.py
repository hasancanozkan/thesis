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
from ROI_function import createROI as roi
from fftFunction import fft
from AnisotropicDiffusionSourceCode import anisodiff as ad
from cv2.ximgproc import guidedFilter
import pywt
from __builtin__ import str
import glob


#mask of fourier
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)


image_list=[]
label_list=[]

for filename in glob.glob('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/*.tif'):
    img= cv2.imread(filename,0)
    label_list.append(img)
    
for filename in glob.glob('C:/Users/oezkan/HasanCan/RawImages/*.tif'):
    img= cv2.imread(filename,0)
    image_list.append(img)


#Wavelet
def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
Indices variabililty of the sample. 
"""
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))
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
    img_roi=roi(img_Eq)
    
    #list of parameters for filters
    param_bl = [[9,11],[100,125,150]]#changed
    param_gf = [[2,4],[0.2,0.4]]
    param_ad = [[10,15],[(0.5,0.5),(1.,1.)]]#changed
    
    #sigma x-y
    param_x = [1.0,1.5,2.0]#changed
    param_y = [1.0,1.5,2.0]#changed
    #setting constants
    param_beta1= [0.2,0.5]
    param_beta2 = [0.125]
    
    bl_class_result = ""
    ad_class_result = ""
    gf_class_result = ""
    wave_class_result = "" 
    
    #apply the filters
    for index in range (0,4,1):
        
        if(index == 0):
            for i in range(len(param_bl[0])):
                for j in range(len(param_bl[1])):
                    #Bilateral
                    img_filtered = cv2.bilateralFilter(img_Eq,param_bl[0][i],param_bl[1][j],param_bl[1][j])
                   
                    #apply frangi for different scales
                    for t in range(len(param_beta1)):
                        for z in range(len(param_beta2)):
                            v = []
                            for k in range(len(param_x)):
                                for l in range(len(param_y)):
                                        v.append(img_as_ubyte(frangi(img_filtered,sigma_x = param_x[k],sigma_y = param_y[l],beta1=param_beta1[t],beta2= param_beta2[z],  black_ridges=True))) 
                                         
                                

                           
                            img_fr = v[0]
                            for sigma_index in range(len(v)-1):
                                img_fr = np.minimum(img_fr,v[sigma_index+1])
                            img_fr = img_fr*img_roi
                            #I will try another filetering here
                            kernel=np.ones(5,np.uint8)
                            min_img = cv2.morphologyEx(img_fr,cv2.MORPH_OPEN,kernel)
                            min_img = guidedFilter(min_img, min_img, 2, 0.2)                                                
                            ''' this part of the code applies old algorithm to get rid of unwanted shapes
                            may be I can delete this part later
                            '''
                            
                            
                            #img_fr.astype(np.float32)
                            _, img_thresh = cv2.threshold(min_img,5,255,cv2.THRESH_BINARY)
                            """
                            # y-axes
                            kernel_y = np.ones((8,1),np.uint8)
                            img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)
 
                            # x-axes
                            kernel_x = np.ones((1,8),np.uint8)
                            img_morph2 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_x)
                                
                            img_morph = img_morph1 + img_morph2
                            #plt.imshow(img_morph,'gray'),plt.title('morph')
                            #plt.show()"""
                            _,contours,_ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            defects = []
                            defectImage = np.zeros((img_thresh.shape))

                            for index in range(len(contours)):
                                if(cv2.contourArea(contours[index]) > 100):
                                    defects.append(contours[index])
 
                                    defectImage = np.zeros((img_thresh.shape))

                                    cv2.drawContours(defectImage, defects, -1, 1, -1)
                                    defectImage = img_as_ubyte(defectImage)
                             
                            #check f! score of all possibilities
                            print classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1])))
                            print "bl_"+str(param_bl[0][i])+"_"+str(param_bl[1][j])+ "_x_"   +str(param_x[k])+"_y_"+str(param_y[l])+ "_beta1_" + str(param_beta1[t]) + "_beta2_" + str(param_beta2[z])
                                   
                            plt.subplot(1,3,1),plt.imshow(img_fr,'gray'),plt.title('img_fr')
                            plt.subplot(1,3,2),plt.imshow(min_img,'gray'),plt.title('min_img')
                            plt.subplot(1,3,3),plt.imshow(defectImage,'gray'),plt.title('img_thresh')

                            plt.show()
                                                                    
                            '''if you dont apply upper part
                            defectImage should be removed from classification result
                            ''' 
                            
                            
            
print(time.time() - start_time1)
