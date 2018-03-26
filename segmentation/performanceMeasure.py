'''
Created on 26.03.2018

@author: oezkan
'''
import numpy as np

def perf_measure(truth, result):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(result)): 
        for j in range(len(result)):
            if np.any(truth[i][j])==np.any(result[i][j])==1:
                TP += 1
            if np.any(result[i][j])==1 and np.any(truth[i][j])!=np.any(result[i][j]):
                FP += 1
            if np.any(truth[i][j])==np.any(result[i][j])==0:
                TN += 1
            if np.any(result[i][j])==0 and np.any(truth[i][j])!=np.any(result[i][j]):
                FN += 1

    return(TP, FP, TN, FN)
