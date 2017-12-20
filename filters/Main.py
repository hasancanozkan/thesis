import numpy as np
#import skimage.util
import matplotlib.pyplot as plt
from PIL import Image

import BilateralFilter2D as bilat
import StructureTensor2D as ST
import Vesselness2D as Vessel
import GuidedFilter as gf
import time


# Load image 'coronaries.jpg' as grayscale image and convert it to
# a float32 scale with grayvalues ranging from [0,1]
image = Image.open('crack.tif').convert("L")
image = np.asarray(image)
image = image.astype(np.float32) / 255.0
image = image[500:1500,500:1500]
#image = skimage.util.random_noise(image, mode='gaussian', var = 0.0005)
#plt.imsave("coronaries_noisy.jpg",image,cmap='gray')
#Bilateral Filter Test
#bilatResult = bilat.bilateralFilter(image, 3 ,40.0, 0.1)

start = time.time()
bilatResult = gf.guided_Filter(image,image,5,0.3**2)
stop = time.time()
print(stop-start)

#fig = plt.figure("Structure Tensor")
fig = plt.figure("Bilateral Filter")
plt.subplot(1,3,1);plt.imshow(image, 'gray')
plt.title('Original Image')
plt.subplot(1,3,2);plt.imshow(bilatResult, 'gray')
plt.title('Guided Filtered Image')
plt.subplot(1,3,3);plt.imshow(np.abs(image-bilatResult), 'gray')
plt.title('Guided Difference Image')
plt.show()
#Call the function calculateStructureTensor(image,s) with s = 0.5
flat, edge, corner = ST.calculateStructureTensor(bilatResult, 0.5)

#Create a Figure showing the original image and the three created results
fig = plt.figure("Structure Tensor")
plt.subplot(2,2,1)
plt.title('Original')
plt.xticks([]) 
plt.yticks([])
plt.imshow(bilatResult,cmap='gray')
plt.subplot(2,2,2)
plt.title('Flat Patch')
plt.xticks([]) 
plt.yticks([])
plt.imshow(flat,cmap='gray',vmin=0, vmax=1)
plt.subplot(2,2,3)
plt.title('Edge')
plt.xticks([]) 
plt.yticks([])
plt.imshow(edge,cmap='gray',vmin=0, vmax=1)
plt.subplot(2,2,4)
plt.title('Corner')
plt.xticks([]) 
plt.yticks([])
plt.imshow(corner,cmap='gray',vmin=0, vmax=1)
plt.show()

#Calculate four Vesselness images with sigma = [1, 1.5, 2, 3]
vesselness_s_1 = Vessel.calculateVesselness2D(bilatResult, 1)
vesselness_s_15 = Vessel.calculateVesselness2D(bilatResult, 1.5)
vesselness_s_2 = Vessel.calculateVesselness2D(bilatResult, 2)
vesselness_s_3 = Vessel.calculateVesselness2D(bilatResult, 3)

result = Vessel.getHighestVesselness(vesselness_s_1,vesselness_s_15,vesselness_s_2,vesselness_s_3)

#Create a new Figure (subplots) with six images: the original image, the four calcualted vesselness
#images and the combined result.
fig = plt.figure('Vesselness')
plt.subplot(2,3,1)
plt.title('Original')
plt.xticks([]) 
plt.yticks([])
plt.imshow(bilatResult,cmap='gray',vmin=0, vmax=np.max(image))
plt.subplot(2,3,2)
plt.title('sigma = 1')
plt.xticks([]) 
plt.yticks([])
plt.imshow(vesselness_s_1,cmap='gray',vmin=0, vmax=np.max(image))
plt.subplot(2,3,3)
plt.title('sigma = 1.5')
plt.xticks([]) 
plt.yticks([])
plt.imshow(vesselness_s_15,cmap='gray',vmin=0, vmax=np.max(image))
plt.subplot(2,3,4)
plt.title('Result')
plt.xticks([]) 
plt.yticks([])
plt.imshow(result,cmap='gray',vmin=0, vmax=np.max(image))
plt.subplot(2,3,5)
plt.title('sigma = 2')
plt.xticks([]) 
plt.yticks([])
plt.imshow(vesselness_s_2,cmap='gray',vmin=0,vmax=np.max(image))
plt.subplot(2,3,6)
plt.title('sigma = 3')
plt.xticks([]) 
plt.yticks([])
plt.imshow(vesselness_s_3,cmap='gray',vmin=0, vmax=np.max(image))
plt.show()