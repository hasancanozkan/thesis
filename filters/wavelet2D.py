'''
Created on 25.12.2017

@author: oezkan
'''
import pywt
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from PIL import Image

img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
#img_nocrack1 = np.asarray(img_nocrack1)

"""
start_time1 = time.time()
wp = pywt.WaveletPacket2D(data=img_nocrack1, wavelet='haar', mode='symmetric')
print(time.time() - start_time1)
"""
"""
#pywt.dwt2(data, wavelet, mode='symmetric', axes=(-2, -1))
coeffs = pywt.dwt2(img_nocrack1, 'haar') # tuple
cA, (cH,cV,cD) = coeffs
"""
start_time1 = time.time()
#2D multilevel decomposition
coeffs = pywt.wavedec2(img_nocrack1, wavelet='haar', level=4) 

#this is how pywt library shows to make coefficients zero but doesnt work
#coeffs[2] == tuple([np.zeros_like(v) for v in coeffs[2]])
#2D multilevel reconstruction 
recon_img = pywt.waverec2(coeffs, wavelet='haar')
print(time.time() - start_time1)

recon_img=np.uint8(recon_img)
#recon_img_1 = Image.fromarray(recon_img, 'L')

plt.subplot(1,2,1),plt.imshow(img_nocrack1,"gray"),plt.title('Original_nocrack')
plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(recon_img,"gray"),plt.title('crack_1')
plt.xticks([]), plt.yticks([])

plt.show()

