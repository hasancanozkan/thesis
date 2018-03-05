'''
Created on 01.02.2018

@author: oezkan
'''

import cv2
import numpy as np
from random import randint

def createROI(img):
    
    threshold_value = np.min(img) 
    _,ROI = cv2.threshold(img,(threshold_value+30),255,cv2.THRESH_BINARY) #more or less same but
    #_,ROI = cv2.threshold(img,(threshold_value*2),255,cv2.THRESH_BINARY)
    kernel_ROI = np.ones((3,3),np.uint8)
    ROI = cv2.erode(ROI,kernel_ROI,iterations = 1) 


    kernel_x = np.ones((1,8),np.uint8)
    img_morph2 = cv2.morphologyEx(ROI, cv2.MORPH_OPEN, kernel_x)
    
    
    kernel_y = np.ones((8,1),np.uint8)
    img_morph1 = cv2.morphologyEx(ROI, cv2.MORPH_OPEN, kernel_y)

    img_morph = img_morph2 * img_morph1
    return img_morph
    
    

    """
    threshold_value = np.min(img) 
    _,img_thr = cv2.threshold(img,threshold_value+30,255,cv2.THRESH_BINARY)

    blur = cv2.blur(img_thr,(5,5))

    _, blur = cv2.threshold(blur,250,255,cv2.THRESH_BINARY)
    return blur
    """