'''
Created on 03.04.2018

@author: oezkan
'''
from fftFunction import fft
from adaptedFrangi import frangi 
from skimage import img_as_ubyte
import threading
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Vesselness2D import calculateVesselness2D

img_filtered_list = []

mask_fft = cv2.imread('C:/Users/oezkan/eclipse-workspace/thesis/filters/ModQ_EL_Poly-Bereket3.tif',0)
img_raw = cv2.imread('C:/Users/oezkan/HasanCan/RawImages/0000006214_bad_.tif',0)
img_fft = fft(img_raw, mask_fft) 
img_Eq = cv2.equalizeHist(img_fft)

class myThread (threading.Thread):
    def __init__(self, name, img_Eq, param1,param2,sigma_x,sigma_y):
        threading.Thread.__init__(self)
        self.name = name
        self.img_Eq = img_Eq
        self.param_1 = param1
        self.param_2 = param2
        self.sigmas = (sigma_x,sigma_y)
    def run(self):
        bilteralFiltering(self.name,self.img_Eq, self.param_1,self.param_2,sigmas)

# Define a function for the thread
def bilteralFiltering(threadName,img_Eq,param1,param2,sigma):
    img_filtered_list.append(img_as_ubyte(frangi(cv2.bilateralFilter(img_Eq,param1,param2,param2),sigma[0] ,sigma[1] ,0.5,0.125,  True)))

threadLock = threading.Lock()
threads = []

# Create two threads as follows
param1_bl = [15,25]
param2_bl = [100,150]
sigmas=[1.0,1.5]#,2.0]
counter = 0
for param_1 in param1_bl:
    for param_2 in param2_bl:
        for sigma_x in sigmas:
            for sigma_y in sigmas:
                print(param_1,param_2,sigma_x,sigma_y)
                threads.append(myThread("Thread "+str(counter), img_Eq, param_1, param_2,sigma_x,sigma_y))
                counter += 1
start = time.time()
# Start new Threads
for t in threads:
    t.start()

for t in threads:
    t.join()

print(time.time()-start)

plt.subplot(2,2,1)
plt.imshow(np.where(img_filtered_list[0]>0,255,0),'gray')
plt.subplot(2,2,2)
plt.imshow(np.where(img_filtered_list[1]>0,255,0),'gray')
plt.subplot(2,2,3)
plt.imshow(np.where(img_filtered_list[2]>0,255,0),'gray')
plt.subplot(2,2,4)
plt.imshow(np.where(img_filtered_list[4]>0,255,0),'gray')
plt.show()