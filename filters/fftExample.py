'''
Created on 13.01.2018

@author: oezkan
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt


#import time

img = cv2.imread('originalImages/0000331736.tif',0)
img = np.float32(img)

#mask image and normalize
mask = cv2.imread('ModQ_EL_Poly-Bereket3.tif',0)
mask = np.float32(mask)
#mask = cv2.normalize(mask,0,1,cv2.NORM_MINMAX)


dft = cv2.dft(img,flags = cv2.DFT_COMPLEX_OUTPUT)#1204,1024,2L
dft_shift = np.fft.fftshift(dft) # 1024,1024,2L


#magnitude_spectrum = 20*np.log(cv2.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1])) #returns(2L,1024,1024)

fshift_real = dft_shift[:,:,0]*mask #1024,1024
fshift_imaginary = dft_shift[:,:,1]*mask

#print (fshift_real,'end',fshift_imaginary)
#print dft.real#same as dft_shift
#print dft.imag #full of zero
#print fshift.shape
#print dft_shift[0,0,0]
#print dft_shift[0,0,1]
#print dft_shift[0].shape
fshift=np.zeros((1024,1024,2))
fshift[:,:,0] = fshift_real
fshift[:,:,1] = fshift_imaginary
#print fshift
img_back = cv2.idft(fshift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.imshow(img_back,cmap='gray')
plt.show()




"""
#cv2 version
#dft of the image
dft= cv2.dft(img)
real = dft[0]
imaginary = dft[1]

dft_shift_real = np.fft.fftshift(real)
dft_shift_imaginary = np.fft.fftshift(imaginary)

real_mask = dft_shift_real*mask
imaginary_mask = dft_shift_imaginary*mask 
print (real_mask)




#idft_img = np.array([real_mask,imaginary_mask])
#print idft_img.shape
img_back = cv2.idft([real_mask,imaginary_mask])
img_back.shape


plt.imshow(img_back,cmap = 'gray')
plt.show()

"""
"""
dft_shift = np.fft.fftshift([real,imaginary])




#multiply with mask
fshift = dft_shift*mask

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)




plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('filtered'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(mask2, cmap = 'gray')
plt.title('mask'), plt.xticks([]), plt.yticks([])
plt.show()

#cv2.imwrite('mask.tif',mask2)
#cv2.imwrite('input.tif',img)
#cv2.imwrite('normalized.tif',np.uint8(img_back))
"""
"""
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
"""