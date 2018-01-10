'''
Created on 08.01.2018

@author: oezkan
'''
import cv2
import numpy as np
from scipy import signal
import time
#from matplotlib import pyplot as plt
from PIL import Image



def dht2(img):
    # filter kernels for low and high pass filtering
    lp = 1.0 / np.sqrt(2) * np.asarray([[1, 1]])
    hp = 1.0 / np.sqrt(2) * np.asarray([[-1, 1]])

    # rows
    sig = np.copy(img)
    approximation = signal.convolve2d(sig, lp, boundary='wrap')[:,1::2]
    # 'wrap' specifies that the convolution should wrap around if elements
    # of the filter kernel leave the image. This leads to the first and last
    # column in the convoluted image being equal. 
    # We crop the first columns for that reason and sample 
    # down by a factor of 2 
    detail = signal.convolve2d(sig, hp, boundary='wrap')[:,1::2]

    # copy image
    sig = np.copy(approximation)
    approx = signal.convolve2d(sig, lp.T, boundary='wrap')[1::2,:]
    detailH = signal.convolve2d(sig, hp.T, boundary='wrap')[1::2,:]

    sig = np.copy(detail)

    detailV = signal.convolve2d(sig, lp.T, boundary='wrap')[1::2,:]
    detailD = signal.convolve2d(sig, hp.T, boundary='wrap')[1::2,:]

    return [approx, detailH, detailV, detailD]

def idht2(approx, detailH, detailV, detailD):
    lp = 1.0 / np.sqrt(2) * np.asarray([[1, 1]])
    hp = 1.0 / np.sqrt(2) * np.asarray([[1, -1]])

    [h, w] = np.shape(approx)

    # columns
    usapprox = np.zeros((2 * h, w))
    usapprox[1::2,:] = approx

    usdetail = np.zeros((2 * h, w))
    usdetail[1::2,:] = detailH

    approx = signal.convolve2d(usapprox, lp.T, boundary='wrap') + signal.convolve2d(usdetail, hp.T, boundary='wrap')
    approx = approx[1:, :]

    usapprox = np.zeros((2 * h, w))
    usdetail = np.zeros((2 * h, w))
    usapprox[1::2,:] = detailV
    usdetail[1::2,:] = detailD

    detail = signal.convolve2d(usapprox, lp.T, boundary='wrap') + signal.convolve2d(usdetail, hp.T, boundary='wrap')
    detail = detail[1:,:]

    # lines
    usapprox = np.zeros((2 * h, 2 * w))
    usapprox[:,1::2] = approx
    usdetail = np.zeros((2 * h, 2 * w))
    usdetail[:,1::2] = detail

    img = signal.convolve2d(usapprox, lp, boundary='wrap') + signal.convolve2d(usdetail, hp, boundary='wrap')
    img = img[:,1:]

    return img

img = cv2.imread('0000000231_crack.tif', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)
img /= 255.0

[approx, detailH, detailV, detailD] = dht2(img)
img_r = idht2(approx, detailH, detailV, detailD)

#plt.subplot(1,3,1),plt.imshow(approx,"gray"),plt.title('111')

normalizedImg = cv2.normalize(detailH, 0, 255, cv2.NORM_MINMAX)
normalizedImg *=255
normalizedImg = np.uint8(normalizedImg)
cv2.imwrite('detailH.tif',normalizedImg)

"""
aprrox_float = Image.fromarray(approx)
aprrox_float.save("approx_float.tif","TIFF")
approx *= 255
approx=np.uint8(approx)
"""

#plt.subplot(1,3,2),plt.imshow(approx,"gray"),plt.title('222')

#plt.subplot(1,3,3),plt.imshow(detailV,"gray"),plt.title('333')

detailV *= 255
detailV = np.uint8(detailV)
#plt.show()

img_r *=255
img_r = np.uint8(img_r)
"""
cv2.imwrite('haar_approx.tif',approx)
cv2.imwrite('haar_detailV.tif',detailV)
cv2.imwrite('img_r.tif',img_r)"""