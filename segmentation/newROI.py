'''
Created on 16.03.2018

@author: oezkan
'''
from skimage import img_as_ubyte
import cv2
import numpy as np
from matplotlib import pyplot as plt

def createROI(img):


    threshold_value = np.min(img)
    _,ROI = cv2.threshold(img,(threshold_value+30),255,cv2.THRESH_BINARY)
    invertROI = cv2.bitwise_not(ROI)
    
    rect_y = cv2.getStructuringElement(cv2.MORPH_RECT,(3,6))
    rect_x = cv2.getStructuringElement(cv2.MORPH_RECT,(6,3))
    
    rect_opened_y = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_y)
    rect_opened_y = cv2.bitwise_not(rect_opened_y)
    
    rect_opened_x = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_x)
    rect_opened_x = cv2.bitwise_not(rect_opened_x)
    
    rect = rect_opened_x * rect_opened_y
    
    kernel_ROI = np.ones((3,3),np.uint8)
    ROI_Morph = cv2.erode(rect,kernel_ROI,iterations = 1) 
    
    
    kernel_x = np.ones((1,8),np.uint8)
    img_morph2 = cv2.morphologyEx(ROI_Morph, cv2.MORPH_OPEN, kernel_x)
        
        
    kernel_y = np.ones((8,1),np.uint8)
    img_morph1 = cv2.morphologyEx(ROI_Morph, cv2.MORPH_OPEN, kernel_y)
    
    img_morph = img_morph2 * img_morph1

    return img_morph
