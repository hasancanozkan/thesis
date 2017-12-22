import numpy as np

# Define a box filter function to perform the mean filter with box /windows radius of r
# Use the algorithm given in the original paper
# Input Parameters:
#    img: 2D input image
#    r: radius of the box (see paper by He et al.)
# Output
#     output image
def boxfilter(img, r):
    rows, columns = img.shape
    resultImg = np.zeros(img.shape)

    #In the original paper a 1D box filter via moving sum is given
    #!!!!Expand this for the 2D case here!!!!
    #Important: do not use any loop. Ensure the O(n) complexity
    #Hint: you can use numpy.cumsum() to calculate the cumulative sum
    #and use numpy.tile() to repeat an array/matrix
    #for details please see numpy documentation
    
    #cumulative sum of y-axis
    imgCum = np.cumsum(img, 0)
    
    #image border
    resultImg[0 : r+1, :] = imgCum[r : 2*r+1, :]
    #regular
    resultImg[r+1 : rows-r, :] = imgCum[2*r+1 : rows, :] - imgCum[0 : rows-2*r-1, :]
    #image border
    resultImg[rows-r: rows, :] = np.tile(imgCum[rows-1, :], [r, 1]) - imgCum[rows-2*r-1 : rows-r-1, :]

    #cumulative sum of x-axis
    imgCum = np.cumsum(resultImg, 1)
    
    #image border
    resultImg[:, 0 : r+1] = imgCum[:, r : 2*r+1]
    #regular
    resultImg[:, r+1 : columns-r] = imgCum[:, 2*r+1 : columns] - imgCum[:, 0 : columns-2*r-1]
    #image border
    resultImg[:, columns-r: columns] = np.tile(imgCum[:, columns-1], [r, 1]).T - imgCum[:, columns-2*r-1 : columns-r-1]

    return resultImg

# Implementation of guided filter as described in the original paper
# Use the provided algorithm 
# Input Parameters:
#     I: the guidance image
#     p: input image
#     r: radius of the box/window
#     epsilon: epsilon (penalization term)
# Output
#     output image
def guided_Filter(I, p, r, epsilon):

    #Normalization term
    N = boxfilter(np.ones([ I.shape[0], I.shape[1]]), r)

    #Create the filtered image of the guidance and input
    meanI = boxfilter(I,r)/N
    meanP = boxfilter(p,r)/N 
    
    #calculate variance and covariance
    covarianceIp = (boxfilter(I*p,r) / N) - meanI * meanP
    varianceI = (boxfilter(I * I, r)/ N) - meanI * meanI

    #calculate a and b according to the equations
    a = covarianceIp / (varianceI + epsilon)
    b = meanP - a * meanI

    #mean filter a and b
    meanA = boxfilter(a, r)/N
    meanB = boxfilter(b, r)/N

    # q = meanA*I +meanB -> output
    return (meanA * I + meanB)
