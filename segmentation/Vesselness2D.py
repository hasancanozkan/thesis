import numpy as np
from scipy.ndimage.filters import gaussian_filter 
#Initialization of constants
beta = 2*0.5**2

def calculateVesselness2D(image, s):

    image_gauss = gaussian_filter(image, sigma=s)
    
    scaling = 1
    if(type(s) is tuple):
        scaling = ((s[0]+s[1])/2)**2 
    else:
        scaling = s**2

    dy,dx = np.gradient(image_gauss)

    dxy,dx2 = np.gradient(dx)
    dy2= np.gradient(dy,axis= 0)

    lambda2,lambda1 = np.array(eigvals_symm(dx2*scaling,dxy*scaling,dy2*scaling))
    maxMag = np.sqrt(np.max(lambda1)**2 + np.max(lambda2)**2)
    c = 2*(maxMag/4)**2

    lambda1[lambda1 == 0] = 1e-10
    rb = (lambda2 / lambda1) ** 2
    s2 = lambda1 ** 2 + lambda2 ** 2
    
    vesselness = np.exp(-rb / beta) * (np.ones(np.shape(image)) - np.exp(-s2 / c))
    vesselness[lambda1 > 0] = 0

    return vesselness

#l1 > l2
def eigvals_symm(M00, M01, M11):
    l1 = (M00 + M11) / 2 + np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    l2 = (M00 + M11) / 2 - np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    return l1, l2
