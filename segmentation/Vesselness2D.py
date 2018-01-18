import numpy as np
from scipy.ndimage.filters import gaussian_filter 

#Initialization of constants
beta = 0.5
c = 0.08
eps = 0.0000001

#Calculate the Vesselness-Filter from Frangi
#Input parameters: 
#   image: 2-D image
#   s: sigma for gauss filter
#Output parameters: 
#   vesselness: 2-D image
def calculateVesselness2D(image, s):
    # create empty result image
    vesselness = np.zeros(image.shape)
    # gauss filter the input with given sigma
    image_gauss = gaussian_filter(image, sigma=s)
    
    #gradient calculation
    dx,dy = np.gradient(image_gauss)

    # Create omponents of the Hessian Matrix [dx2 dxy][dyx dy2]
    dx2,dxy = np.gradient(dx)
    dyx,dy2 = np.gradient(dy)
    
    #normalization -> multiply the hessian components with sigma^2
    dx2 = dx2*s**2
    dy2 = dy2*s**2
    dxy = dxy*s**2
    dyx = dyx*s**2

    # 1) Get eigenvalues of Hessian for each pixel
    # 2) Sort the eigenvalues ascending such that |ev_1| < |ev_2|
    # 3) Call the vesselnessMeasure(eigenvalues) function and set the
    #    calculated value to the specific pixel
    for x in range(0,dx2.shape[0]):
        for y in range(0,dx2.shape[1]):

            eigenvalues = np.linalg.eigvals(np.array([[dx2[x,y] , dxy[x,y]],[dyx[x,y], dy2[x,y]]],np.float32))

            if(np.abs(eigenvalues[0])>np.abs(eigenvalues[1])):
                buf = eigenvalues[0]
                eigenvalues[0] = eigenvalues[1]
                eigenvalues[1] = buf

            vesselness[x,y] = vesselnessMeasure(eigenvalues)

    # return the resulting vesselness image
    return vesselness

#Calculate the 2-D Vesselness Measure (see Frangi paper or lecture slides)
#Use the provides 'c' and 'b' variable from the initialization
def vesselnessMeasure(eigenvalues):
    if(eigenvalues[1] > 0):
        return 0
    else:
        if(eigenvalues[1] == 0):
            eigenvalues[1] = eps 
        
        RB = eigenvalues[0]/eigenvalues[1]
        S = np.sqrt(eigenvalues[0]**2 + eigenvalues[1]**2)
        return np.exp(-(RB**2)/(2*beta**2))*(1-np.exp(-(S**2)/(2*c**2)))

#Takes four vesselness images and gets sets the max value to a result
def getHighestVesselness(v1,v2,v3,v4):
    return np.maximum(np.maximum(np.maximum(v1,v2),v3),v4)
