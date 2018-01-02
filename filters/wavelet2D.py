'''
Created on 25.12.2017

@author: oezkan
'''
import pywt
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample. 
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))

# load the images
img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_crack2 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000281_crack.tif",0)
img_crack3 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000001220_crack.tif",0)

# convert to float32
img_nocrack1 = np.float32(img_nocrack1)
img_nocrack1 /= 255

start_time1 = time.time()
#2D multilevel decomposition
level = 1
wavelet = 'haar'
coeffs_nocrack1_haar= pywt.wavedec2(img_nocrack1, wavelet=wavelet,level=level) 

#calculate the threshold
sigma = mad(coeffs_nocrack1_haar[-level])
threshold_haar = sigma*np.sqrt( 2*np.log(img_crack1.size)) #this is soft thresholding
#threshold_haar = 0.085
newCoeffs_nocrack1_map_haar = map (lambda x: pywt.threshold(x,threshold_haar,mode='soft'),coeffs_nocrack1_haar)


newCoeffs_nocrack1_map_haar = pywt.waverec2(newCoeffs_nocrack1_map_haar, wavelet=wavelet)
newCoeffs_nocrack1_map_haar *= 255
newCoeffs_nocrack1_map_haar=np.uint8(newCoeffs_nocrack1_map_haar)
print(time.time() - start_time1) 
#save the images
cv2.imwrite('wave_nocrack1_map_haar.tif',newCoeffs_nocrack1_map_haar)