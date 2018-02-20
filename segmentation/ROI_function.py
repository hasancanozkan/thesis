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
    
    '''
        the part of the code below is another way of creating ROI
        I applied blur
    '''
    """img = cv2.equalizeHist(img)


    threshold_value = np.min(img) 
    _,img_thr = cv2.threshold(img,threshold_value+50,255,cv2.THRESH_BINARY)

    blur = cv2.blur(img_thr,(9,9))

    _,contours,_ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    defectImage = np.zeros((img_thr.shape))
    cv2.drawContours(defectImage, contours, -1, 1, -1)
    """
    return img_morph 
""" 
    _, contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for i in range(len(contours)):
            if(cv2.contourArea(contours[i]) > 300):
                defects.append(contours[i])
            
                        
    defectImage = np.zeros((ROI.shape))
    cv2.drawContours(defectImage, defects, -1, 1, -1)

    return img_morph
   """ 
