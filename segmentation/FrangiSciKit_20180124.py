'''
Created on 20.03.2018

@author: oezkan
'''

from skimage.filters import frangi, hessian
import cv2
from matplotlib import pyplot as plt
import time
from PIL import Image
import numpy
import pywt
from newROI import createROI
from fftFunction import fft


wavelet = 'bior2.8' #'bior2.8' #'haar' 'no'
scaleLow  = 0.75
scaleHigh = scaleLow+0.1
scaleStep = 10
beta1     = 0.5
beta2     = 15
dilateSize= 19  # kernel size for dilation (BB mask)
ThFrangi  = 27
BPkersize = 5 
ThBP      = 7

def mysaveImg( arr, filename ):
    tmpimg = []
    tmpimg = Image.fromarray( arr )
    tmpimg.save( filename )


def myBPfilter( arr, kersize):
    blurImg = cv2.blur( arr, (kersize, kersize) )  
    arr = numpy.where( blurImg < arr, 0, arr)
    blurImg =  numpy.where( arr==0, 0, blurImg)
    dif = blurImg  - arr 
    return dif

def myremoveBoundary( myimg, mask_tmp, kersize ):
    if ( kersize > 0):
        kernel = numpy.ones((kersize,kersize),'uint8')
        mask_tmp = cv2.dilate( mask_tmp, kernel )
    myimg = numpy.where( mask_tmp == 255, 0, myimg)
    return myimg   
    
def myInvertarr( arr ): 
    tmparr = arr
    tmparr = numpy.where( arr==0, 255, tmparr )
    tmparr = numpy.where( arr==255, 0, tmparr )
    return tmparr

def myWavelet( img, wavelet ):
    #wavelet = 'haar' 
    #decompose to 2nd level coefficients
    [cA, (cH, cV, cD) ] = pywt.dwt2(img, wavelet=wavelet)
    maxval = cA.max()
    cA = cA * ( img.max() / maxval)
    return cA


imagePath = 'C:/Users/oezkan/HasanCan/RawImages/'
imagePathSave = 'C:/Users/Desktop/results/newResults/'
img = cv2.imread( imagePath + "0000007542_bad_.tif",0)
mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)


img = fft(img, mask_fft) 
    
    # apply histogram equalization
img_Eq = cv2.equalizeHist(img)
    
# mask is the ROI
mask =createROI(img_Eq)
#img  = cv2.pyrDown( img, (512,512))
#mask = cv2.pyrDown(mask, (512,512))

meanVal = img.mean()
img_maskIn = cv2.convertScaleAbs( numpy.where( mask == 0, meanVal, img ) )
mask_inv = cv2.convertScaleAbs( myInvertarr( mask ) )

# ----------------------------------------------------------------------------

if wavelet == 'no':
    img_mask = img_maskIn
else:
    # save standard downscale image as comparison
    cv2.imwrite( imagePathSave+'waveletcmp_downscale.tif', cv2.pyrDown( img_maskIn, (512,512)) )    
    # wavelet
    img_mask = myWavelet( img_maskIn, wavelet )
    cv2.imwrite( imagePathSave+'wavelet_'+wavelet+'.tif', img_mask)


start_time1 = time.time()
img_fr = frangi(img_mask,scale_range=(scaleLow,scaleHigh),scale_step=scaleStep,beta1=beta1,beta2=beta2)
print (time.time()-start_time1)
#img_fr = myremoveBoundary( img_fr, mask_inv, dilateSize )
#mysaveImg( img_fr, imagePathSave+'frangi_low' + str(scaleLow) + "_high" + str(scaleHigh) + "_step" + str(scaleStep) + "_beta1" + str(beta1) +  '.tif')

normFrangi = cv2.convertScaleAbs( img_fr/img_fr.max()*255 )
#mysaveImg( normFrangi, 'frangiNorm_low' + str(scaleLow) + "_high" + str(scaleHigh) + "_step" + str(scaleStep) + "_beta1_" + str(beta1) + "_beta2_" + str(beta2) + '.tif')
ret,FrangiTH = cv2.threshold(normFrangi,ThFrangi,255,cv2.THRESH_BINARY)
#mysaveImg( FrangiTH, imagePathSave+'frangiTh_low' + str(scaleLow) + "_high" + str(scaleHigh) + "_step" + str(scaleStep) + "_beta1_" + str(beta1) + "_beta2_" + str(beta2) + '.tif')

im_BP = myBPfilter( img_mask, BPkersize)
#im_BP = myremoveBoundary( im_BP, mask_inv, dilateSize )
ret,BPTH = cv2.threshold(im_BP,ThBP,255,cv2.THRESH_BINARY)
#mysaveImg( BPTH, imagePathSave+'BPTh_ksize' + str(BPkersize) + '_Th' + str(ThBP) +'.tif' )

start_time1 = time.time()
img_hs = hessian(img_mask, scale_range=(scaleLow,scaleHigh),scale_step=scaleStep,beta1=beta1)
print (time.time()-start_time1)

im_BP = numpy.where(im_BP>0,1,0)
img_fr = numpy.where(img_fr>0.6,1,0)

plt.subplot(1,3,1),plt.imshow(img_mask,"gray"),plt.title('img-mask')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(img_fr,"gray"),plt.title('frangi')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(im_BP,"gray"),plt.title('bandpass')
plt.xticks([]), plt.yticks([])
plt.show()

