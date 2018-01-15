'''
Created on 13.01.2018

@author: oezkan
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
from cv2 import dft
#import time

img = cv2.imread('originalImages/0000598257.tif',0)
mask = cv2.imread('ModQ_EL_Poly-Bereket3.tif',0)
mask /= 255
#mask = cv2.resize(mask,None,fx=2, fy=2)

#numpy version
dft = np.fft.fftshift(np.fft.fft2(img))
dft_shift = np.fft.fftshift(dft)
#magnitude = 20 * np.log(1 + np.abs(dft_shift))
print dft_shift.shape
dft_result = dft_shift*mask

image_result = np.fft.ifftshift(dft_result)
image_result =  np.abs(np.fft.ifft2(image_result))
print image_result.shape 
print(image_result)#now it is not complex
difference = np.abs(image_result-img)
plt.imshow(difference,'gray')
plt.show()
