'''
Created on 11.01.2018

@author: oezkan
'''
import cv2
import time
import pywt
import numpy as np

img_nocrack1 = cv2.imread("C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000000831_Keincrack.tif",0)
img_crack1 = cv2.imread("C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000000231_crack.tif",0)
img_crack2 = cv2.imread("C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000000281_crack.tif",0)
img_crack3 = cv2.imread("C:/Users/oezkan/eclipse-workspace/thesis/filters/originalImages/0000001220_crack.tif",0)

""" # first wavelet then bilateral
# convert to float32
img = np.float32(img_crack1)
img /= 255.0


#2D multilevel decomposition
level = 2
wavelet = 'haar'
 
start_time1 = time.time()
#decompose to 2nd level coefficients
[cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)] =  pywt.wavedec2(img, wavelet=wavelet,level=level)
coeffs = [cA2,(cH2, cV2, cD2)]

#reconstruction
recon_img = pywt.waverec2(coeffs, wavelet=wavelet)

# normalization to convert uint8
normalizedImg = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
normalizedImg *=255
normalizedImg = np.uint8(normalizedImg)

normalizedImg = cv2.bilateralFilter(normalizedImg,9,75,75)
print(time.time() - start_time1)
#save images
cv2.imwrite('combined.tif',normalizedImg)
""" 
img_crack1 = cv2.bilateralFilter(img_crack1,9,75,75)

img = np.float32(img_crack1)
img /= 255.0


#2D multilevel decomposition
level = 2
wavelet = 'haar'
 
start_time1 = time.time()
#decompose to 2nd level coefficients
[cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)] =  pywt.wavedec2(img, wavelet=wavelet,level=level)
coeffs = [cA2,(cH2, cV2, cD2)]

#reconstruction
recon_img = pywt.waverec2(coeffs, wavelet=wavelet)

# normalization to convert uint8
normalizedImg = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
normalizedImg *=255
normalizedImg = np.uint8(normalizedImg)
print(time.time() - start_time1)
cv2.imwrite('combined2.tif',normalizedImg)