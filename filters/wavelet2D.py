'''
Created on 25.12.2017

@author: oezkan
'''
import pywt
import numpy as np
import cv2
import time
#from PIL import Image # use to show float32 images
#from matplotlib import pyplot as plt


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample. 
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))

# load the images 
img_1MB = cv2.imread("originalImages/0000689998.tif",0)
img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_crack2 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000281_crack.tif",0)
img_crack3 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000001220_crack.tif",0)
print np.max(img_1MB)
# convert to float32  there is no change if we dont do this!!
#img = np.float32(img_crack1)
img = (img_1MB/255.0)
print np.max(img)

#2D multilevel decomposition
level = 2
wavelet = 'haar'
 
#start_time1 = time.time()
#decompose to 2nd level coefficients
[cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)] =  pywt.wavedec2(img_1MB, wavelet=wavelet,level=level)
coeffs = [cA2,(cH2, cV2, cD2)]


#calculate the threshold
sigma = mad(coeffs[-level])
#sigma = 75
threshold = sigma*np.sqrt( 2*np.log(img_crack1.size/2)) #this is soft thresholding
#print threshold
#threshold = 50
newCoeffs = map (lambda x: pywt.threshold(x,threshold,mode='hard'),coeffs)


#reconstruction
#recon_img = pywt.waverec2(newCoeffs, wavelet=wavelet)
recon_img= pywt.waverec2(coeffs, wavelet=wavelet)
print np.max(recon_img)

# normalization to convert uint8
normalizedImg = cv2.normalize(recon_img, 0, 255, cv2.NORM_MINMAX)
normalizedImg *=255
#print(time.time() - start_time1)
print np.max(normalizedImg)

"""
#show the chosen image
plt.imshow(normalizedImg,"gray"),plt.title('111')
plt.show()
"""
#save the chosen image
cv2.imwrite('approx_crack1_hard1.tif', normalizedImg)


"""
to save as float32
aprox= Image.fromarray(normalizedImg)
aprox.save("normalizedImg.tif","TIFF")

#plt.imshow(aprrox_float32,"gray"),plt.title('111')
#plt.show()
"""