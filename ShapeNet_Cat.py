# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:10:34 2019

Iterates through each KorNet condition and provide top-5 labels

@author: VAYZENB
"""

import os
os.chdir('C:/Users/vayzenb/Desktop/GitHub Repos/KorNet/')

import numpy as np
import pandas as pd
from itertools import chain
from ShapeNet import load_model
from ShapeNet import image_loader
from ShapeNet import predict

cond = ['Outline', 'Pert', 'IC',  'Outline_Black', 'Pert_Black', 'IC_Black', 'Outline_Black_Filled', 'Pert_Black_Filled']
#cond = ['Outline', 'Pert', 'IC']
suf = ['', '_ripple', '_IC', '', '_ripple', '_IC','', '_ripple']
#suf = ['', '_ripple', '_IC', ]
#Load ImageNet and KorNet classes
IN=pd.read_csv('IN_Classes.csv', sep=',',header=None).to_numpy()
KN=pd.read_csv('KN_Classes.csv', sep=',',header=None).to_numpy() 


model = load_model() 
model.eval() #Set model into evaluation mode

#Loop through each condition
for ii in range(0, len(cond)):
    CNN_Labels = np.empty((len(KN), 6), dtype = object)
    CNN_Labels[:,0] = KN[:,0]    
    
    #Loop through each image in a condition and predict class label
    for kk in range(0, len(KN)):
        imFile = 'Stim/Test/' + cond[ii] + '/OBJ (' + str(KN[kk,1]) + ')' + suf[ii] +'.png'
        IM = image_loader(imFile)

        out = predict(IM, model)
        out = out.numpy()
        out = out.tolist()
        out = list(chain(*out))
                
        CNN_Labels[kk,1:] = [IN[out[0]][0], IN[out[1]][0], IN[out[2]][0], IN[out[3]][0], IN[out[4]][0]]
        
    np.savetxt('Results/' + cond[ii] + "_PredClasses.csv", CNN_Labels, delimiter=',', fmt= '%s')

    

