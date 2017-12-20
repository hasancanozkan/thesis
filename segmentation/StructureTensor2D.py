import numpy as np 
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
#init
thresh = 0.0003

#Structure Tensor calculation
#Input parameters: 
#   image: 2-D image
#   s: sigma for gauss filter
#Output parameters: 
#   flat_areas: 2-D image
#   straight_edges: 2-D image
#   corners: 2-D image
def calculateStructureTensor(image, s):
    
    #create empty result images
    flat_areas = np.zeros(image.shape)
    straight_edges = np.zeros(image.shape)
    corners = np.zeros(image.shape)
    
    #Gauss filter the image with the provided sigma
    image_gauss = gaussian_filter(image, s)
   
    #calculate gradients in x and y direction
    dx,dy = np.gradient(image_gauss)

    #Gauss filtering of the Structure Tensor components
    K_dy2 = gaussian_filter(dy*dy,s)
    K_dx2 = gaussian_filter(dx*dx,s)
    K_dxy = gaussian_filter(dx*dy,s)

    # 1) Get eigenvalues of Structure Tensor for each pixel
    # 2) Sort the eigenvalues ascending such that ev1 < ev2
    # 3) Set the pixels to '1' for the three result images
    #    Use the provided threshold 'thresh' for the eigenvalue comparison
    for x in range(0,K_dxy.shape[0]):
        for y in range(0,K_dxy.shape[1]):
            H = np.array([[K_dx2[x,y] , K_dxy[x,y]],[K_dxy[x,y], K_dy2[x,y]]],np.float32)
            eigenvalues = np.linalg.eigvals(H)
            #sorting
            if(np.abs(eigenvalues[1]) > np.abs(eigenvalues[0])):
                buf = eigenvalues[1]
                eigenvalues[1] = eigenvalues[0]
                eigenvalues[0] = buf
                
            if eigenvalues[0] < thresh and eigenvalues[1] < thresh:
                flat_areas[x,y] = 1
            elif eigenvalues[0] >= thresh and eigenvalues[1] < thresh:
                straight_edges[x,y] = 1
            else:
                corners[x,y] = 1
    # return the three resulting images
    return flat_areas, straight_edges, corners
