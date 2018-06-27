'''
Created on 22.03.2018

@author: oezkan
'''
from adaptedFrangiSelfBeta2 import frangi 
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
from newROI import createROI

from __builtin__ import str
import glob

#mask of fourier
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006214_bad_.tif',0)
img_label = cv2.imread('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/0000006214_bad_.tif',0)

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
    img_back1 = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    
     
    plt.subplot(1,2,1),plt.imshow(img_back,'gray')
    plt.subplot(1,2,2),plt.imshow(img_back1,'gray')
    plt.show()
    
    
    #!!!!!! may be I wont need to convert 8 bit integer here
    normalizedImg = cv2.normalize(img_back, 0, 255, cv2.NORM_MINMAX)
    normalizedImg *=255
    normalizedImg = np.uint8(normalizedImg)
    
    return normalizedImg

f = np.fft.fft2(img_raw)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img_raw, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()



"""
image_list=[]
label_list=[]


for filename in glob.glob('C:/Users/oezkan/HasanCan/AnnotatedImages_255crack/*.tif'):
    img= cv2.imread(filename,0)
    label_list.append(img)
    
for filename in glob.glob('C:/Users/oezkan/HasanCan/RawImages/*.tif'):
    img= cv2.imread(filename,0)
    image_list.append(img)




#start_time1 = time.time()
    
for im in range(len(image_list)):    
    # labeled signal
    _,labeled_crack = cv2.threshold(label_list[im],245,255,cv2.THRESH_BINARY)
    kernel_label = np.ones((1,1),np.uint8)   
    labeled_crack = cv2.dilate(labeled_crack,kernel_label,iterations =1)
    
    # apply fft for grid fingers
    img_fft = fft(image_list[im], mask_fft) 
    
    # apply histogram equalization
    #img_Eq = cv2.equalizeHist(img_fft)
    
    #img_roi=createROI(img_Eq)
    
    #img_res = img_fft*img_roi
    
    #cv2.imwrite(str(im)+'.tif',img_res)"""