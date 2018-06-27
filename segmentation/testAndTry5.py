'''
Created on 21.03.2018

@author: oezkan
'''
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
from performanceMeasure import perf_measure
from newROI import createROI
from fftFunction import fft
from __builtin__ import str
import glob
from skimage.io import imsave
from AnisotropicDiffusionSourceCode import anisodiff as ad
from cv2.ximgproc import guidedFilter
import pywt


# arr is the filtered image
def myBPfilter( arr, kersize):
    blurImg = cv2.blur( arr, (kersize, kersize) )  
    arr = np.where( blurImg < arr, 0, arr)
    blurImg =  np.where( arr==0, 0, blurImg)
    dif = blurImg  - arr 
    return dif


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
        
        class_result = ""
        thres= [5]
        kernel = [21]
        
        
        for i in range(len(thres)):
            onlyFPvec = []
            sensitivityVec=[]
            for j in range(len(kernel)):
                
                img_filtered = cv2.bilateralFilter(img_fft,5,100,100)
                #img_filtered = ad(img_fft, niter=5, kappa=50,gamma=0.10, option=1)
                #img_filtered=np.uint8(img_filtered)# if not frangi does not accept img_filtered because it is float between -1 and 1
                #img_filtered = guidedFilter(img_fft, img_fft, 2, 0.2**2)
                #[cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)]  =  pywt.wavedec2(img_fft, wavelet='haar',level=2)
                #coeffs = [cA2,(cH2, cV2, cD2)]
                #reconstruction
                #recon_img= pywt.waverec2(coeffs, wavelet='haar')


                # normalization 
                #img_filtered = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
                #plt.imshow(img_filtered*255,'gray')
                #plt.show()
                start_time1 = time.time()
                img_bp = myBPfilter(img_filtered, kernel[j])
                print(time.time() - start_time1)                       
                
                #cv2.imwrite(str(im)+'_img_BP_'+'_thresh_'+str(thres[i])+'kernel_'+str(kernel[j])+'.tif', img_bp)
                _, img_thresh = cv2.threshold(img_bp,thres[i],255,cv2.THRESH_BINARY) 
                #img_roi = cv2.resize(img_roi,(512,512))
                #plt.imshow(img_roi,'gray')
                #plt.show()
                img_fr_roi = img_thresh*img_roi 
                """
                perform_result = sensitivity, _, _, _, onlyFP = perf_measure(labeled_crack, img_fr_roi, img_roi)
                class_result = class_result+"_x 1.0_n_1.5_thresh_0.20"+"\n" + str(perform_result)+"\n"

                #save the data
                with open(str(im)+'_BP_classification_results.txt','w') as output:
                    output.write(str(class_result))
                """