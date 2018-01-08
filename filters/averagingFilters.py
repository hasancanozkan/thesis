'''
Created on 08.01.2018

@author: oezkan
'''
import cv2
import time
from matplotlib import pyplot as plt

img = cv2.imread('0000000831_Keincrack.tif', 0)

start_time1 = time.time()
mean_img = cv2.blur(img,(2,2))
print(time.time() - start_time1)

start_time1 = time.time()
median_img = cv2.medianBlur(img,7)
print(time.time() - start_time1)

"""
plt.subplot(131),plt.imshow(img,"gray"),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(mean_img,"gray"),plt.title('mean')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(median_img,"gray"),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()
"""

#save images
#cv2.imwrite('mean_2_nocrack_1.tif',mean_img)
cv2.imwrite('median_7_nocrack_1.tif',median_img)