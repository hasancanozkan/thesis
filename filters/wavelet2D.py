'''
Created on 25.12.2017

@author: oezkan
'''
import pywt
import numpy as np
import cv2
import time
#from PIL import Image # use to show float32 images
from matplotlib import pyplot as plt

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample. 
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))

# load the images 
img_1MB = cv2.imread("originalImages/0000689998.tif",0)
img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_crack2 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000281_crack.tif",0)
img_crack3 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000001220_crack.tif",0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006214_bad_.tif',0)
img_vessel = cv2.imread('C:/Users/oezkan/Downloads/healthy/01_h.jpg',0)

# convert to float32  there is no change if we dont do this!!
#img = np.float32(img_crack1)
#img_raw = (img_raw/255.0)

level = 2
wavelet = 'haar'
#decompose to 2nd level coefficients
start_time1 = time.time()
[cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)]  =  pywt.wavedec2(img_vessel, wavelet=wavelet,level=level)
coeffs = [cA2,(cH2, cV2, cD2)]
#reconstruction
recon_img= pywt.waverec2(coeffs, wavelet=wavelet)
print(time.time() - start_time1)


# normalization 
img_filtered = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)

#plt.subplot(1,2,1),plt.imshow(img_raw,'gray')
#plt.subplot(1,2,2),plt.imshow(img_filtered*255,'gray'),

#plt.show()
cv2.imwrite('Wave_.tif',img_filtered*255)


"""
#calculate the threshold
sigma = mad(coeffs[-level])
#sigma = 75
threshold = sigma*np.sqrt( 2*np.log(img_crack1.size/2))
#print threshold
#threshold = 50
newCoeffs = map (lambda x: pywt.threshold(x,threshold,mode='hard'),coeffs)
"""

