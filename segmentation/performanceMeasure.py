'''
Created on 26.03.2018

@author: oezkan
'''
import numpy as np
'''
Be careful that truth and result values should be 0-255!!
'''
def perf_measure(truth, result, img_roi):
    
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    # Number of Crack-Pixel
    NCP = 0.0
    # Total Number of Pixels in ROI
    TNP = 0.0
    
    if np.any(truth)==1:
        for i in range(len(result)): 
            for j in range(len(result)):
                if truth[i][j]==255:
                    NCP +=1
        print 'NCP: ' + str(NCP)
                
        for i in range(len(result)): 
            for j in range(len(result)):
                if(img_roi[i][j] == 1):
                    if truth[i][j]==result[i][j]==255:
                        TP += 1
                    elif result[i][j]==255 and truth[i][j]!=result[i][j]:
                        FP += 1
                    elif truth[i][j]==result[i][j]==0:
                        TN += 1
                    elif result[i][j]==0 and truth[i][j]!=result[i][j]:
                        FN += 1
                TNP += 1
        print 'TNP: ' + str(TNP)
        print 'FP:' + str(FP)
 
        sensitivity = (TP / (TP+FN))
        specificity = (TN / (TN+FP))
        FPR = 1 - specificity
        FNR = 1 - sensitivity
        onlyFP =  FP /( TNP - NCP )
        
    else :
        
        for i in range(len(result)): 
            for j in range(len(result)):
                if(img_roi[i][j] == 1):
                    TNP +=1
                    if truth[i][j]==result[i][j]==0:
                        TP += 1
                        FN += 1
                    elif result[i][j]==255 and truth[i][j]!=result[i][j]:
                        FP += 1
                        FN += 1
        sensitivity = specificity= (TP / (TP+FN))
        FPR = FNR = 1 - specificity
        onlyFP = FP / TNP 
        print 'TNP: ' + str(TNP)
        print 'FP:' + str(FP)

    return( sensitivity, specificity , FPR, FNR, onlyFP)
