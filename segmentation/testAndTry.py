'''
Created on Jan 23, 2018

@author: HasanCan
'''
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter 
from adaptedFrangi import frangi


img_raw =  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000689998.tif',0)
img_raw2=  cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736.tif',0)
img_raw3 = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006604_bad_.tif',0)

mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
labeled_crack = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000689998_CrackLabel.tif')
labeled_crack2 = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000331736_CrackLabel.tif')
labeled_crack3 = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000006604_bad_.tif')


blur = gaussian_filter(img_raw, (1,15))
blur2 =gaussian_filter(img_raw, (15,1)) 

img_fr = frangi(img_raw,sigma_x=2,sigma_y=2,beta1=0.5,beta2= 0.05,  black_ridges=True)



plt.subplot(2,3,1),plt.imshow(img_raw,'gray'),plt.title('img_raw')
plt.subplot(2,3,2),plt.imshow(blur,'gray'),plt.title('blur')
plt.subplot(2,3,3),plt.imshow(img_fr,'gray'),plt.title('img_fr')


plt.show()

