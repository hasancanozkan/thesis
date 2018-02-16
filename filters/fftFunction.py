'''
Created on 07.02.2018

@author: oezkan
'''

import numpy as np
import cv2

def fft(img,mask):
    
    
    #convert the image to float32
    img = np.float32(img)
    #convert the mask to float32
    mask = np.float32(mask)
    mask = cv2.normalize(mask,0,1,cv2.NORM_MINMAX)
    
    # apply fft
    dft = cv2.dft(img,flags = cv2.DFT_COMPLEX_OUTPUT)#1204,1024,2L
    dft_shift = np.fft.fftshift(dft) # 1024,1024,2L
    
    fshift_real = dft_shift[:,:,0]*mask #1024,1024
    fshift_imaginary = dft_shift[:,:,1]*mask
    
    fshift=np.zeros((len(mask),len(mask),2))
    fshift[:,:,0] = fshift_imaginary
    fshift[:,:,1] = fshift_real 
    
    #apply inverse fft
    img_back = cv2.idft(fshift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    
    #!!!!!! may be I wont need to convert 8 bit integer here
    normalizedImg = cv2.normalize(img_back, 0, 255, cv2.NORM_MINMAX)
    normalizedImg *=255
    normalizedImg = np.uint8(normalizedImg)
    
    return normalizedImg