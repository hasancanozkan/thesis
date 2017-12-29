'''
Created on 25.12.2017

@author: oezkan
'''
import pywt
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from numpy import imag

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


image = np.float32(img_nocrack1)
image /= 255

start_time1 = time.time()
#2D multilevel decomposition
level = 1
coeffs= pywt.wavedec2(image, wavelet='haar',level=level) 

#calculate the threshold
sigma = mad(coeffs[-level])
threshold = sigma*np.sqrt( 2*np.log(image.size)) 
newCoeffs = map (lambda x: pywt.threshold(x,threshold,mode='soft'),coeffs)


recon_img = pywt.waverec2(newCoeffs, wavelet='haar')
recon_img *= 255
recon_img=np.uint8(recon_img)

print(time.time() - start_time1)

cv2.imwrite('wavepic.tif',recon_img)