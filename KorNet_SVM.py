# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:42:19 2020

@author: vayze
"""

import numpy as np
import pandas as pd
from sklearn import svm
import random
from itertools import chain

cond = ['Outline', 'Pert', 'IC']
modelType = ['CorNet-S_1','CorNet-S_2', 'CorNet-S_3', 'CorNet-S_4', 'CorNet-S_5', 'FF_IN', 'R_IN', 'FF_SN', 'R_SN']

KN=pd.read_csv('KN_Classes.csv', sep=',',header=None).to_numpy()

trK = 20 #Number of training images to use
folK = 10 #Number of folds over the training set


train_labels = [np.repeat(1, trK).tolist(), np.repeat(2, trK).tolist()]
train_labels = list(chain(*train_labels))
test_labels = [1,2]

for cc in range(0, len(cond)):
    CNN_Acc = np.empty((1890, 5), dtype = object)    
    n=0
    for mm in range(0, len(modelType)):
        tsVec = np.load('Activations/Test/' + modelType[mm] + '_' + cond[cc] + '_acts.npy')     
        
        for kk in range(0, len(KN)):
            trVec1 = np.load('Activations/Train/' + modelType[mm] + '_' + KN[kk][0] + '_acts.npy')
            
            for jj in range(kk+1, len(KN)):
                    #pull second image                       
                    if KN[kk][2] == KN[jj][2]: #Check if they are matched for animacy   
                        trVec2 = np.load('Activations/Train/' + modelType[mm] + '_' + KN[jj][0] + '_acts.npy')
                        test_vec = np.zeros((2, len(tsVec[0,:])))
                        test_vec[0] = tsVec[kk,:]
                        test_vec[1] = tsVec[jj,:]
                        
                        tempScore = 0
                        for fl in range(0,folK): #loop through folds
                            train_vec = np.append(trVec1[random.sample(range(0,len(trVec1)), trK),:], trVec2[random.sample(range(0,len(trVec2)), trK),:],axis =0)
                            
                            clf = svm.SVC(kernel='linear', C=1).fit(train_vec, train_labels)
                            tempScore = tempScore + clf.score(test_vec, test_labels)
                            
                        CNN_Acc[n,0] = modelType[mm]
                        CNN_Acc[n,1] = cond[cc]
                        CNN_Acc[n,2] = KN[kk][0]
                        CNN_Acc[n,3] = KN[jj][0]
                        CNN_Acc[n,4] = tempScore/folK
                        print(modelType[mm], cond[cc], KN[kk][0], KN[jj][0],tempScore/folK)
                        
                        n = n + 1
                    else:
                        continue
                    
    np.savetxt('Results/' + cond[cc] + '_SVM_allModels.csv', CNN_Acc, delimiter=',', fmt= '%s')
                            
                            
                            
                        
    