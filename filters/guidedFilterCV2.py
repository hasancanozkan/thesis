'''
Created on 22.12.2017

For some reason the filter does not affect on image 
maybe it is not compatible with python 2.7, the sourcecode was for Python 3.0._
@author: oezkan
'''
import cv2
from cv2.ximgproc import guidedFilter
from matplotlib import pyplot as plt
import time


img_nocrack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000831_Keincrack.tif",0)
img_crack1 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000231_crack.tif",0)
img_crack2 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000000281_crack.tif",0)
img_crack3 = cv2.imread("D://oezkan/Data/MASTERTHESIS_EL_start/0000001220_crack.tif",0)

start_time1 = time.time()
#bl_nocrack1 = cv2.bilateralFilter(img_nocrack1,5,75,75)
gf_nocrack1 = guidedFilter(img_crack1, img_crack1, 16, 0.09)
print(time.time() - start_time1)

print gf_nocrack1[150,150]
print img_crack1[150,150]

plt.subplot(1,2,1),plt.imshow(img_crack1,"gray"),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(gf_nocrack1,"gray"),plt.title('Guided')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.imwrite('gf_nocrack1.tif',gf_nocrack1)