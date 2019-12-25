# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:10:34 2019

@author: VAYZENB
"""

import os
os.chdir('C:/Users/vayzenb/Desktop/GitHub Repos/KorNet/')

import numpy as np
import pandas as pd
import itertools
import glob
import random
from itertools import chain
from sklearn import svm
from ShapeNet import load_model
from ShapeNet import image_loader
from ShapeNet import get_vector

#cond = ['Outline', 'Pert', 'IC',  'Outline_Black', 'Pert_Black', 'IC_Black', 'Outline_Black_Filled', 'Pert_Black_Filled']
cond = ['Outline', 'Pert', 'IC']
suf = ['', '_ripple', '_IC', '', '_ripple', '_IC','', '_ripple']

KN=pd.read_csv('KN_Classes.csv', sep=',',header=None).to_numpy() 

layer = "avgpool"
model = load_model() 
model.eval() #Set model into evaluation mode

IMName = "Stim/Training/All/Dog/Dog_4 (2).jpg"

vec = get_vector(IMName, model, layer).numpy()

train_labels = [np.repeat(1, 20).tolist(), np.repeat(2, 20).tolist()]
train_labels = list(chain(*train_labels))
test_labels = [1,2]

trK = 20 #Number of training images to use
folK = 3 #Number of folds over the training set

for ii in range(0, len(cond)): 
    AllActs = {"Train" : np.zeros((trK*2, 2048)), "Test" : np.zeros((2, 2048))}
    
    #Loop through each possible image combination in a condition 
    for kk in range(0, len(KN)):
        #load first image
        IM1 = "Stim/Test/" + cond[ii] + "/Obj (" + str(KN[kk][1]) + ")" + suf[ii] + ".png"
        AllActs["Test"][0] = get_vector(IM1, model, layer).numpy() #Extract image vector
        #load training list
        imList1 = [os.path.basename(x) for x in glob.glob("Stim/Training/" + KN[kk][0] + "/*.jpg")] #pull image category list
        
        for jj in range(kk+1, len(KN)):
            #pull second image                       
            if KN[kk][2] == KN[jj][2]: #Check if they are matched for animacy              
                
                IM2 = "Stim/Test/" + cond[ii] + "/Obj (" + str(KN[jj][1]) + ")" + suf[ii] + ".png"    
                AllActs["Test"][1] = get_vector(IM2, model, layer).numpy()
                imList2 = [os.path.basename(x) for x in glob.glob("Stim/Training/" + KN[jj][0] + "/*.jpg")]
                
                for fl in range(0,folK): #loop through folds
                    #randomize order of image lists every iteration
                    imList1 = random.sample(imList1, len(imList1))
                    imList2 = random.sample(imList2, len(imList2))
                    for tr in range(0,trK): #loop through training images
                        trIM1= "Stim/Training/" + KN[kk][0] + "/" + imList1[tr]
                        trIM2= "Stim/Training/" + KN[jj][0] + "/" + imList2[tr]
                        
                        #Extract features for each training image
                        AllActs["Train"][tr] = get_vector(trIM1, model, layer).numpy()
                        AllActs["Train"][tr+20] = get_vector(trIM2, model, layer).numpy()
                        
                    #Run SVM
                    clf = svm.SVC(kernel='linear', C=1).fit(AllActs["Train"], train_labels)
                    clf.score(AllActs["Test"], test_labels)
                    
                        
                        
                
            else: #move to next iteration
                continue
            
            
        

