'''
Created on Jan 23, 2018

@author: HasanCan
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

start_time1 = time.time()

img_raw =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736.tif',0)
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
labeled_crack = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736_CrackLabel.tif')
maskRoi = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736_BusLabel.tif',0) 

# labeled signal 
_, labeled_crack = cv2.threshold(labeled_crack[:,:,2],127,255,cv2.THRESH_BINARY)
kernel_label = np.ones((2,2),np.uint8)
labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)

#apply mask for busbars and ROI, for now artificially drawn
# !!! it can not work as I do now !!!! I have to change convolve it with mask

'''be sure that image is either img_raw or white labelede maskRoi
'''
# apply fft for grid fingers
img_fft = fft(img_raw, mask_fft) 

# apply histogram equalization
img_Eq = cv2.equalizeHist(img_fft)

'''this part is my own roi, which is bad with star cracks
then multiply roi with tghe result of frangi
'''
img_roi=roi(img_Eq)

#list of parameters for filters
param_bl = [[5,9,11,13,15],[25,50,75,100,125,150]]
param_gf = [[2,3,4,5,6,7],[0.1,0.2,0.3,0.4]]
param_ad = [[3,5,10,15,20],[(0.5,0.5),(1.,1.),(2.,2.),(5.,5.),(10.,10.)]]

#param_scale = [(0.5,0.6),(1.0,1.1),(1.5,1.6),(2.0,2.1),(2.5,2.6),(3.0,3.6),(3.5,3.6),(4.0,4.1)] #0.5 or even 1 is very low
param_scale = [(1.5,1.6),(2.0,2.1),(2.5,2.6),(3.0,3.6),(3.5,3.6),(4.0,4.1),(4.5,4.6),(5.0,5.1)]


bl_class_result = ""
ad_class_result = ""
gf_class_result = ""
wave_class_result = "" 

#apply the filters
for i in range (0,4,1):
    
    if(i == 0):
        v = [[],[],[],[],[],[],[],[]]
        for i in range(len(param_bl[0])):
            for j in range(len(param_bl[1])):
                print i,'end',j
                #Bilateral
                img_filtered = cv2.bilateralFilter(img_Eq,param_bl[0][i],param_bl[1][j],param_bl[1][j])

                #apply frangi for different scales
                for k in range(len(param_scale)):
                    v[k] = frangi(img_filtered,scale_range=param_scale[k],scale_step=1,beta1=0.5,beta2= 0.05,  black_ridges=True)
                    v[k] = img_as_ubyte(v[k])

                '''if you dont want to take roi remove from the code below
                ''' 
                min_img = np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])*img_roi
                
                ''' this part of the code applies old algorithm to get rid of unwanted shapes
                may be I can delete this part later
                '''
                
                #img_fr.astype(np.float32)
                _, img_thresh = cv2.threshold(min_img,20,255,cv2.THRESH_BINARY)
     
                # y-axes
                kernel_y = np.ones((10,1),np.uint8)
                img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)
     
                # x-axes
                kernel_x = np.ones((1,10),np.uint8)
                img_morph2 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_x)
  
                img_morph = img_morph1 + img_morph2
    
                _,contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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
                new_result = (classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1]))))
                bl_class_result = bl_class_result+"bl_i_"+str(i)+"_j_"+str(j)+new_result
                #save the data
                with open('bl_classification_results.txt','w') as output:
                    output.write(bl_class_result)
    if(i == 1):
        #Anisotropic
        v = [[],[],[],[],[],[],[],[]]
        for i in range(len(param_ad[0])):
            for j in range(len(param_ad[1])):
                img_filtered = ad(img_Eq, niter=param_ad[0][i],step=param_ad[1][j], kappa=50,gamma=0.10, option=1)
                img_filtered=np.uint8(img_filtered)# if not frangi does not accept img_filtered because it is float between -1 and 1

                #apply frangi for different scales
                for k in range(len(param_scale)):
                    v[k] = frangi(img_filtered,scale_range=param_scale[k],scale_step=1,beta1=0.5,beta2= 0.05,  black_ridges=True)
                    v[k] = img_as_ubyte(v[k])
                '''if you dont want to take roi remove from the code below
                '''    
                min_img = np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])*img_roi
                ''' this part of the code applies old algorithm to get rid of unwanted shapes
                may be I can delete this part later
                '''
                
                #img_fr.astype(np.float32)
                _, img_thresh = cv2.threshold(min_img,20,255,cv2.THRESH_BINARY)
     
                # y-axes
                kernel_y = np.ones((10,1),np.uint8)
                img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)
     
                # x-axes
                kernel_x = np.ones((1,10),np.uint8)
                img_morph2 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_x)
  
                img_morph = img_morph1 + img_morph2
    
                _,contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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
                new_result = (classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1]))))   
                ad_class_result = ad_class_result+"ad_i_"+str(i)+"_j_"+str(j)+new_result
                #save the data
                with open('ad_classification_results.txt','w') as output:
                    output.write(ad_class_result)
    if(i == 2):
        #GUIDED
        v = [[],[],[],[],[],[],[],[]]
        for i in range(len(param_gf[0])):
            for j in range(len(param_gf[1])):
                img_filtered = guidedFilter(img_Eq, img_Eq, param_gf[0][i], param_gf[1][j])
                
                #apply frangi for different scales
                for k in range(len(param_scale)):
                    v[k] = frangi(img_filtered,scale_range=param_scale[k],scale_step=1,beta1=0.5,beta2= 0.05,  black_ridges=True)
                    v[k] = img_as_ubyte(v[k])
                    
                '''if you dont want to take roi remove from the code below
                ''' 
                min_img = np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])*img_roi
                
                ''' this part of the code applies old algorithm to get rid of unwanted shapes
                may be I can delete this part later
                '''
                
                #img_fr.astype(np.float32)
                _, img_thresh = cv2.threshold(min_img,20,255,cv2.THRESH_BINARY)
     
                # y-axes
                kernel_y = np.ones((10,1),np.uint8)
                img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)
     
                # x-axes
                kernel_x = np.ones((1,10),np.uint8)
                img_morph2 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_x)
  
                img_morph = img_morph1 + img_morph2
    
                _,contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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
                new_result = (classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1]))))    
                gf_class_result = gf_class_result+"gf_i_"+str(i)+"_j_"+str(j)+new_result
                #save the data
                with open('gf_classification_results.txt','w') as output:
                    output.write(gf_class_result)
    if (i == 3):
        #Wavelet
        def mad(arr):
            """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample. 
        """
            arr = np.ma.array(arr).compressed()
            med = np.median(arr)
            return np.median(np.abs(arr - med))
        img_Eq /=255
        level = 2
        wavelet = 'haar'
        #decompose to 2nd level coefficients
        coeffs =  pywt.wavedec2(img_Eq, wavelet=wavelet,level=level)
        v = [[],[],[],[],[],[],[],[]]
        #calculate the threshold
        sigma = mad(coeffs[-level])
        threshold = sigma*np.sqrt( 2*np.log(img_Eq.size/2)) #this is soft thresholding
        #threshold = 50
        newCoeffs = map (lambda x: pywt.threshold(x,threshold,mode='hard'),coeffs)

        #reconstruction
        recon_img= pywt.waverec2(coeffs, wavelet=wavelet)

        # normalization to convert uint8
        img_filtered = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
        img_filtered *=255
        #apply frangi for different scales
        for k in range(len(param_scale)):
            v[k] = frangi(img_filtered,scale_range=param_scale[k],scale_step=1,beta1=0.5,beta2= 0.05,  black_ridges=True)
            v[k] = img_as_ubyte(v[k])

        '''if you dont want to take roi remove from the code below
                ''' 
        min_img = np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])*img_roi
        
        ''' this part of the code applies old algorithm to get rid of unwanted shapes
                may be I can delete this part later
                '''
                
        #img_fr.astype(np.float32)
        _, img_thresh = cv2.threshold(min_img,20,255,cv2.THRESH_BINARY)
     
        # y-axes
        kernel_y = np.ones((10,1),np.uint8)
        img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)
     
        # x-axes
        kernel_x = np.ones((1,10),np.uint8)
        img_morph2 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_x)
  
        img_morph = img_morph1 + img_morph2
    
        _,contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        new_result = (classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1]))))
                            
        wave_class_result = wave_class_result + new_result
        #save the data
        with open('wave_classification_results.txt','w') as output:
            output.write(wave_class_result)                

print(time.time() - start_time1)

'''
!!!!!!!!!!!!!!!!COMMENTTT!!!!!!!!!!!!
    the rest is to see what I have tried to see better classification results
    it does not use the white labels that I create.
    It basically has the old system with minor changes --> thresholding and morphology
    I need thresholding because the results are inbetween 0-255, but the labeled image are only 0 and 255.
    After showing Daniel, either I will use this or delete it....
'''
"""
img_raw =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000689998.tif',0)
img_raw2=  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736.tif',0)
img_raw3 = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006604_bad_.tif',0)

mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
labeled_crack = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000689998_CrackLabel.tif')
labeled_crack2 = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736_CrackLabel.tif')
labeled_crack3 = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000006604_bad_.tif')

maskRoi = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000689998_BusLabel.tif',0) I wont use this now


_,maskRoi = cv2.threshold(maskRoi,253,255,cv2.THRESH_BINARY)



_, labeled_crack = cv2.threshold(labeled_crack3[:,:,2],127,255,cv2.THRESH_BINARY)
plt.imshow(labeled_crack,'gray'),plt.show()
kernel_label = np.ones((2,2),np.uint8)
labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)

# apply fft for grid fingers
img_fft = fft(img_raw3, mask_fft)

# apply histogram equalization
img_Eq = cv2.equalizeHist(img_fft)

img_roi=roi(img_Eq)


img_filtered = cv2.bilateralFilter(img_Eq,9,75,75)

''' param_scale i arttirdim
'''
param_scale = [(1.5,1.6),(2.0,2.1),(2.5,2.6),(3.0,3.6),(3.5,3.6),(4.0,4.1),(4.5,4.6),(5.0,5.1)]

v = [[],[],[],[],[],[],[],[]]

for k in range(len(param_scale)):
    v[k] = frangi(img_filtered,scale_range=param_scale[k],scale_step=1,beta1=0.5,beta2= 0.05)
    v[k] = img_as_ubyte(v[k])

min_img = np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])*img_roi
max_img  = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(v[6],v[7]),v[5]),v[4]),v[3]),v[2]),v[1]),v[0])
#_, min_img = cv2.threshold(min_img,15,255,cv2.THRESH_BINARY)

img_fr = frangi(img_filtered,scale_range=(2,2.1),scale_step=0.5,beta1=0.5,beta2= 0.05)*img_roi
_,img_fr_thres = cv2.threshold(img_fr,5,255,cv2.THRESH_BINARY)

img_fr = img_as_ubyte(img_fr)

#img_fr.astype(np.float32)
_, img_thresh = cv2.threshold(min_img,20,255,cv2.THRESH_BINARY)
     
    # y-axes
kernel_y = np.ones((10,1),np.uint8)
img_morph1 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_y)
     
    # x-axes
kernel_x = np.ones((1,10),np.uint8)
img_morph2 = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_x)
  
img_morph = img_morph1 + img_morph2
    
_,contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
defects = []
for i in range(len(contours)):
    if(cv2.contourArea(contours[i]) > 200):
        defects.append(contours[i])
     
defectImage = np.zeros((img_thresh.shape))
cv2.drawContours(defectImage, defects, -1, 1, -1)
defectImage = img_as_ubyte(defectImage)

print classification_report(labeled_crack.reshape((labeled_crack.shape[0]*labeled_crack.shape[1])),defectImage.reshape((defectImage.shape[0]*defectImage.shape[1])))


plt.subplot(2,4,1),plt.imshow(labeled_crack3,'gray'),plt.title('labeled')
plt.subplot(2,4,2),plt.imshow(min_img,'gray'),plt.title('min_img')
plt.subplot(2,4,3),plt.imshow(defectImage,'gray'),plt.title('defectImage')
plt.subplot(2,4,4),plt.imshow(img_fr,'gray'),plt.title('img_fr')


plt.show()
"""