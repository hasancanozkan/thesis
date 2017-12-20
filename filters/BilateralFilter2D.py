import numpy as np

def computeGeometricCloseness(i, j, x, y,sigma_d):
    return np.exp(- 0.5 * np.power((computeEuclidianDistance(i,j,x,y) / sigma_d), 2))
    
def computeEuclidianDistance(i, j, x, y):
    return np.sqrt(np.power(i-x ,2) + np.power((j-y),2))
   
def computeIntensityDistance(img, i, j, x, y):
    x_val = img[y,x]
    xi_val = img[j,i]
    return np.abs(x_val - xi_val)

def computePhotometricDistance(img, i, j, x, y,sigma_r):
    return np.exp(-0.5 * np.power(computeIntensityDistance(img, i ,j,x,y) / sigma_r, 2));

def bilateralFilterHelper (img, x, y, width, sigma_d, sigma_r):
    sumWeight = 0
    sumFilter = 0
    # No filtering at the image boundaries;
    if ((x < (width/2)) or (x+(width/2)+1 >= img.shape[1]) or (y < (width/2)) or (y+(width/2)+1 >= img.shape[0])):
        sumWeight = 1
        sumFilter = img[y,x]
    else:
        for i in range(int(x-(width/2)),int(x+(width/2)+1),1):
            for j in range(int(y-(width/2)),int(y+(width/2)+1),1):
                currentWeight = computePhotometricDistance(img, i, j, x, y,sigma_r) * computeGeometricCloseness(i,j,x,y,sigma_d);
                sumWeight += currentWeight
                sumFilter += currentWeight * img[j,i]
           
    return sumFilter / sumWeight;

def bilateralFilter(img, width, sigma_d, sigma_r):
    result = img.copy()
    for y in range(0,img.shape[1],1):
        for x in range(0,img.shape[0],1):         
            result[x,y] = bilateralFilterHelper(img, y, x,width, sigma_d, sigma_r)
    return result
   
        