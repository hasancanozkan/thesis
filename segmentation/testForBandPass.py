'''
Created on 21.03.2018

@author: oezkan
'''
from adaptedFrangi import frangi 
from skimage import img_as_ubyte
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report
from newROI import createROI, createROIfromOriginal
from fftFunction import fft
from AnisotropicDiffusionSourceCode import anisodiff as ad
from cv2.ximgproc import guidedFilter
import pywt
from __builtin__ import str
import glob

# arr is the filtered image
def myBPfilter( arr, kersize):
    blurImg = cv2.blur( arr, (kersize, kersize) )  
    arr = np.where( blurImg < arr, 0, arr)
    blurImg =  np.where( arr==0, 0, blurImg)
    dif = blurImg  - arr 
    return dif

'''
    Next will be as I did before 
'''