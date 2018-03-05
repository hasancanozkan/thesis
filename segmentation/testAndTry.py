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


img_raw1 = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000007361_bad_.tif',0)
labeled_crack1 = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000007361_bad_.tif',0)

#mask of fourier
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)


image_list=[img_raw1]
label_list=[labeled_crack1]

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
    
    img_fft = fft(image_list[im], mask_fft) 
    
    img_Eq = cv2.equalizeHist(img_fft)
    
    img_roi=roi(img_Eq)
    #sigma x-y
    param_x = [1.0]#changed
    param_y = [1.0]#changed
    #setting constants
    param_beta1= [0.2]
    param_beta2 = [0.125]
    
    
    #apply the filters
    for index in range (0,4,1):
        
        if (index == 3):
            #Wavelet
            
            #img_fft /=255
            level = 2
            wavelet = 'haar'
            #decompose to 2nd level coefficients
            coeffs =  pywt.wavedec2(img_Eq, wavelet=wavelet,level=level)
            
            #calculate the threshold
            sigma = mad(coeffs[-level])
            threshold = sigma*np.sqrt( 2*np.log(img_fft.size/2)) #this is soft thresholding
            print threshold
            #threshold = 50
            newCoeffs = map (lambda x: pywt.threshold(x,threshold*2,mode='hard'),coeffs)
            
            #reconstruction
            recon_img= pywt.waverec2(coeffs, wavelet=wavelet)
            plt.imshow(recon_img,'gray'),plt.title('img_recon')
            plt.show()
            # normalization to convert uint8
            img_filtered = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
            img_filtered = recon_img*255
            plt.imshow(img_filtered,'gray'),plt.title('img filtered')
            plt.show()
            
            #print img_filtered.shape
            
            #img_filtered = cv2.cvtColor(img_filtered,cv2.COLOR_BGR2GRAY)
            

            v = [[],[]]
            #apply frangi for different scales
            for k in range(len(param_x)):
                            for l in range(len(param_y)):
                                for t in range(len(param_beta1)):
                                    for z in range(len(param_beta2)):
                                        v[z] = frangi(img_filtered,sigma_x = param_x[k],sigma_y = param_y[l],beta1=param_beta1[t],beta2= param_beta2[z],  black_ridges=True)
                                        v[z] = img_as_ubyte(v[z])
    
                                    min_img = v[0]*img_roi
                                    plt.imshow(min_img,'gray'),plt.title('img min_img')
                                    plt.show()
                
                                    ''' this part of the code applies old algorithm to get rid of unwanted shapes
                                    may be I can delete this part later
                                    '''
                                    
                                    #img_fr.astype(np.float32)
                                    _, img_thresh = cv2.threshold(min_img,3,255,cv2.THRESH_BINARY)
                            
                                    # y-axes
                                    kernel_y = np.ones((8,1),np.uint8)
                                    img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)
         
                                    # x-axes
                                    kernel_x = np.ones((1,8),np.uint8)
                                    img_morph2 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_x)
                                        
                                    img_morph = img_morph1 + img_morph2
        
                                    _,contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                    defectImage = np.zeros((img_thresh.shape))
    
                                    defects = []
                                    for index in range(len(contours)):
                                        if(cv2.contourArea(contours[index]) > 200):
                                            defects.append(contours[index])
         
                                            defectImage = np.zeros((img_thresh.shape))
    
                                            cv2.drawContours(defectImage, defects, -1, 1, -1)
                                            defectImage = img_as_ubyte(defectImage)
                                    
                                                                      
                                    '''if you dont apply upper part
                                    defectImage should be removed from classification result
                                    '''
                                    #check f! score of all possibilities
                                    print classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1])))
                                    #save the data
                                    plt.subplot(2,2,1),plt.imshow(recon_img,'gray'),plt.title('recon_img')
                                    plt.subplot(2,2,2),plt.imshow(img_filtered,'gray'),plt.title('img_filtered')
                                    plt.subplot(2,2,3),plt.imshow(min_img,'gray'),plt.title('min_img')
                                    plt.subplot(2,2,4),plt.imshow(defectImage,'gray'),plt.title('defect')
                                    plt.show()
