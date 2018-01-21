'''
Created on 13.01.2018

@author: oezkan
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


#import time

img = cv2.imread('originalImages/0000331736.tif',0)
img = np.float32(img)

#mask image and normalize
mask = cv2.imread('ModQ_EL_Poly-Bereket3.tif',0)
mask = np.float32(mask)
mask = cv2.normalize(mask,0,1,cv2.NORM_MINMAX)

start_time1 = time.time()
dft = cv2.dft(img,flags = cv2.DFT_COMPLEX_OUTPUT)#1204,1024,2L
dft_shift = np.fft.fftshift(dft) # 1024,1024,2L


#magnitude_spectrum = 20*np.log(cv2.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1])) #returns(2L,1024,1024)

fshift_real = dft_shift[:,:,0]*mask #1024,1024
fshift_imaginary = dft_shift[:,:,1]*mask


fshift=np.zeros((1024,1024,2))
fshift[:,:,0] = fshift_imaginary
fshift[:,:,1] = fshift_real 

#inverse transform
img_back = cv2.idft(fshift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
print(time.time() - start_time1)

normalizedImg = cv2.normalize(img_back, 0, 255, cv2.NORM_MINMAX)
normalizedImg *=255
normalizedImg = np.uint8(normalizedImg)
#save image
#cv2.imwrite('fft.tif',normalizedImg)
#show image
plt.subplot(1,3,1),plt.imshow(normalizedImg,cmap='gray'),plt.title('normalized')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(img_back,cmap='gray'),plt.title('img_back')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(cv2.imread('originalImages/0000331736.tif',0),cmap='gray'),plt.title('img')
plt.xticks([]), plt.yticks([])
plt.show()