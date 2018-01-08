'''
Created on 25.12.2017

@author: oezkan
'''
import pywt
import numpy as np
import cv2
import time
from PIL import Image
from matplotlib import pyplot as plt


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample. 
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))
# scale function
def scale_image(input_image_path,
                output_image_path,
                width=None,
                height=None
                ):
    original_image = Image.open(input_image_path)
    w, h = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=w, height=h))
 
    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')
 
    original_image.thumbnail(max_size, Image.ANTIALIAS)
    original_image.save(output_image_path)
 
    scaled_image = Image.open(output_image_path)
    width, height = scaled_image.size
    print('The scaled image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))

# load the images 
img_1MB = cv2.imread("0000598257.tif",0)
img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_crack2 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000281_crack.tif",0)
img_crack3 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000001220_crack.tif",0)

# convert to float32
img = np.float32(img_nocrack1)
##img /= 255

#2D multilevel decomposition
level = 2
wavelet = 'haar'
 
""" # 3rd level coefficients
[cA3, (cH3, cV3, cD3),(cH2, cV2, cD2), (cH1, cV1, cD1)] =  pywt.wavedec2(img, wavelet=wavelet,level=level)
"""
#2nd level coefficients
[cA2,(cH2, cV2, cD2), (cH1, cV1, cD1)] =  pywt.wavedec2(img, wavelet=wavelet,level=level)
coeffs = [cA2,(cH2, cV2, cD2)]

#zero_coeffs = np.zeros((256, 256))
"""
#calculate the threshold
sigma = mad(coeffs[-level])


threshold_haar = sigma*np.sqrt( 2*np.log(img.size)) #this is soft thresholding

#threshold_haar = 0.085
newCoeffs_nocrack1_map_haar = map (lambda x: pywt.threshold(x,threshold_haar,mode='soft'),coeffs)
newCoeffs_nocrack1_map_haar = pywt.waverec2(newCoeffs_nocrack1_map_haar, wavelet=wavelet)
"""
"""
coeffs= pywt.wavedec2(img, wavelet=wavelet,level=level)

#calculate the threshold
sigma = mad(coeffs[-level])
threshold_haar = sigma*np.sqrt( 2*np.log(img_crack1.size)) #this is soft thresholding
newCoeffs = map (lambda x: pywt.threshold(x,threshold_haar,mode='soft'),coeffs)
"""

recon_img = pywt.waverec2(coeffs, wavelet=wavelet)
plt.subplot(1,3,1),plt.imshow(recon_img,"gray"),plt.title('111')
cv2.imwrite('recon255.tif',recon_img)



#recon_img *= 255
plt.subplot(1,3,2),plt.imshow(recon_img,"gray"),plt.title('222')

recon_img=np.uint8(recon_img)
plt.subplot(1,3,3),plt.imshow(recon_img,"gray"),plt.title('333')
plt.show()
#save the images


#down-scaling the image
#scaled_recon = scale_image('recon.tif', 'scaled_recon.tif', 1024, 1024)
