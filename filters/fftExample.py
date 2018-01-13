'''
Created on 13.01.2018

@author: oezkan
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
from cv2 import dft
import time

img = cv2.imread('originalImages/0000000231_crack.tif',0)

#first fft

# convert float
img = np.float32(img)
#start_time1 = time.time()
dft = cv2.dft(img,flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
#print(time.time() - start_time1)
"""
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
"""

#second inverse fft, ifft

rows, cols = img.shape
crow,ccol = rows/2 , cols/2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-1024:crow+1024,ccol-1024:ccol+1024] = 1

#apply mask and inverse dft
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

"""
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()"""

normalizedImg = cv2.normalize(img_back, 0, 255, cv2.NORM_MINMAX)
normalizedImg *=255
normalizedImg = np.uint8(normalizedImg)
cv2.imwrite('fft_image_1024.tif',normalizedImg)
