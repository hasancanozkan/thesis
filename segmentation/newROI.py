'''
Created on 16.03.2018

@author: oezkan
'''
from skimage import img_as_ubyte
import cv2
import numpy as np
from matplotlib import pyplot as plt
from fftFunction import fft

def mainBorder(img_Eq):
    '''
    This function gets main border which are
    thick bus bars and in rare cases thin bus bars
    It needs to be applied on an image which has histogram equalization!!!!!!!!!!!!
    '''
    threshold_value = np.min(img_Eq)
    _,ROI = cv2.threshold(img_Eq,(threshold_value+60),255,cv2.THRESH_BINARY)
    invertROI = cv2.bitwise_not(ROI)
    
    #setting rectangle kernels
    rect_y = cv2.getStructuringElement(cv2.MORPH_RECT,(4,1000))
    rect_x = cv2.getStructuringElement(cv2.MORPH_RECT,(1000,4))
    
    rect_opened_y = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_y)
    rect_opened_y = cv2.bitwise_not(rect_opened_y)
    
    rect_opened_x = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_x)
    rect_opened_x = cv2.bitwise_not(rect_opened_x)
    
    rect = rect_opened_x * rect_opened_y
    
    kernel_ROI = np.ones((3,3),np.uint8)
    ROI_Morph = cv2.erode(rect,kernel_ROI,iterations = 1) 
    
    
    kernel_x = np.ones((1,8),np.uint8)
    img_morph2 = cv2.morphologyEx(rect, cv2.MORPH_OPEN, kernel_x)
        
        
    kernel_y = np.ones((8,1),np.uint8)
    img_morph1 = cv2.morphologyEx(rect, cv2.MORPH_OPEN, kernel_y)
    
    img_morph = img_morph2 * img_morph1
    return img_morph


def createROI(img_Eq):
    '''
    This function gets main border and 
    bus bars together.
    It needs to be applied on an image which has histogram equalization!!!!!!!!!!!!
    '''

    threshold_value = np.min(img_Eq)
    _,ROI = cv2.threshold(img_Eq,(threshold_value+110),255,cv2.THRESH_BINARY)
    invertROI = cv2.bitwise_not(ROI)
    
    rect_y = cv2.getStructuringElement(cv2.MORPH_RECT,(4,600))
    rect_x = cv2.getStructuringElement(cv2.MORPH_RECT,(600,4))
    
    rect_opened_y = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_y)
    rect_opened_y = cv2.bitwise_not(rect_opened_y)
    
    rect_opened_x = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_x)
    rect_opened_x = cv2.bitwise_not(rect_opened_x)
    
    rect = rect_opened_x * rect_opened_y
    
    kernel_ROI = np.ones((4,4),np.uint8)
    ROI_Morph = cv2.erode(rect,kernel_ROI,iterations = 2) 
    
    
    kernel_x = np.ones((1,10),np.uint8)
    img_morph2 = cv2.morphologyEx(ROI_Morph, cv2.MORPH_OPEN, kernel_x)
        
        
    kernel_y = np.ones((10,1),np.uint8)
    img_morph1 = cv2.morphologyEx(ROI_Morph, cv2.MORPH_OPEN, kernel_y)
    
    img_morph = img_morph2 * img_morph1
    #plt.subplot(1,2,1),plt.imshow(rect,'gray'),plt.title('rect')
    #plt.subplot(1,2,2),plt.imshow(img_morph,'gray'),plt.title('img_morph')

    #plt.show()
    return img_morph
def createROIfromOriginal(img_fft):
    '''
    This function gets main border and 
    bus bars together.
    It needs to be applied on an image which has histogram equalization!!!!!!!!!!!!
    '''

    threshold_value = np.min(img_fft)
    _,ROI = cv2.threshold(img_fft,(threshold_value+55),255,cv2.THRESH_BINARY)
    invertROI = cv2.bitwise_not(ROI)
    
    rect_y = cv2.getStructuringElement(cv2.MORPH_RECT,(4,600))
    rect_x = cv2.getStructuringElement(cv2.MORPH_RECT,(600,4))
    
    rect_opened_y = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_y)
    rect_opened_y = cv2.bitwise_not(rect_opened_y)
    
    rect_opened_x = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_x)
    rect_opened_x = cv2.bitwise_not(rect_opened_x)
    
    rect = rect_opened_x * rect_opened_y
    
    kernel_ROI = np.ones((4,4),np.uint8)
    ROI_Morph = cv2.erode(rect,kernel_ROI,iterations = 2) 
    
    
    kernel_x = np.ones((1,10),np.uint8)
    img_morph2 = cv2.morphologyEx(ROI_Morph, cv2.MORPH_OPEN, kernel_x)
        
        
    kernel_y = np.ones((10,1),np.uint8)
    img_morph1 = cv2.morphologyEx(ROI_Morph, cv2.MORPH_OPEN, kernel_y)
    
    img_morph = img_morph2 * img_morph1
    #plt.subplot(1,2,1),plt.imshow(rect,'gray'),plt.title('rect')
    #plt.subplot(1,2,2),plt.imshow(img_morph,'gray'),plt.title('img_morph')

    #plt.show()
    return img_morph

def adaptiveROI(img_fft):
    '''
    it returns the adaptive thresholded image for an fft filtered image
    could be used at the end to improve the crack trace
    '''
    img_thresh = cv2.adaptiveThreshold(img_fft,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    img_thresh = cv2.bitwise_not(img_thresh)
    
    return img_thresh

