'''
Created on Jan 23, 2018

@author: HasanCan
'''
from skimage.filters import frangi, hessian
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import time


img = cv2.imread('C:/Users/HasanCan/Downloads/000-filteredImage.tif',0)
mask = cv2.imread('C:/Users/HasanCan/Downloads/201-ErodeMask.tif',0)

start_time1 = time.time()
img_fr = frangi(img,scale_range=(2,5),scale_step=2,beta1=0.5)
print (time.time()-start_time1)

start_time1 = time.time()
img_hs = hessian(img, scale_range=(2,5),scale_step=2,beta1=0.5)
print (time.time()-start_time1)

plt.subplot(1,2,1),plt.imshow(img_fr,"gray"),plt.title('frangi')
plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(img_hs,"gray"),plt.title('hessian')
plt.xticks([]), plt.yticks([])
plt.show()