# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:10:34 2019

@author: VAYZENB
"""

import os
os.chdir('C:/Users/vayze/Desktop/GitHub Repos/KorNet/')

import numpy as np
import pandas as pd
import itertools
import glob
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.models
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image
from itertools import chain
from sklearn import svm


#cond = ['Outline', 'Pert', 'IC',  'Outline_Black', 'Outline_Black_Filled', 'Pert_Black_Filled']
cond = ['Outline', 'Pert', 'IC']

#suf = ['', '_ripple', '_IC', '', '', '_ripple', '_IC','', '_ripple']
suf = ['', '_ripple', '_IC']

#FF = Feedforward; R = Recurrent
#IN = imagenet trained; SN = shape trained
#models = ['FF_IN', 'R_IN', 'FF_SN', 'R_SN']
models = ['R_SN']

KN=pd.read_csv('KN_Classes.csv', sep=',',header=None).to_numpy() 

device = torch.device("cuda")
trK = 20 #Number of training images to use
folK = 5 #Number of folds over the training set

train_labels = [np.repeat(1, trK).tolist(), np.repeat(2, trK).tolist()]
train_labels = list(chain(*train_labels))
test_labels = [1,2]

scaler = T.Resize((224, 224))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = T.ToTensor()
#Set image loader for model
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert("RGB")
    image = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0))
    return image     

for mm in range(0, len(models)):
    
    #select model to run
    if models[mm] == 'FF_IN':
        model = torchvision.models.alexnet(pretrained=True)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
        model.classifier = new_classifier #replace model classifier with stripped version
        layer = "fc7"
        actNum = 4096
        
    elif models[mm] == 'R_IN':
        model = torchvision.models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        layer = "avgpool"
        actNum = 2048
                
    elif models[mm] == 'FF_SN':
        model = torchvision.models.alexnet(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        checkpoint = torch.load('ShapeNet_AlexNet_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
        model.classifier = new_classifier #replace model classifier with stripped version
        #model.to(device)
        layer = "fc7"
        actNum = 4096
        
    elif models[mm] == 'R_SN':
        model = torchvision.models.resnet50(pretrained=False)
        #model = torch.nn.DataParallel(model.features)
        checkpoint = torch.load('ShapeNet_ResNet50_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        model = nn.Sequential(*list(model.children())[:-1])
        #model.to(device)
        layer = "avgpool"
        actNum = 2048
        
    model.eval() #Set model into evaluation mode
    
    CNN_Acc = np.empty((210, len(cond) + 2), dtype = object)
    for ii in range(0, len(cond)): 
        AllActs = {"Train" : np.zeros((trK*2, actNum)), "Test" : np.zeros((2, actNum))}
        n = 0
        #Loop through each possible image combination in a condition 
        for kk in range(0, len(KN)):
            #load first image
            IM1 = image_loader("Stim/Test/" + cond[ii] + "/Obj (" + str(KN[kk][1]) + ")" + suf[ii] + ".png")
            vec = model(IM1).detach().numpy() #Extract image vector
            AllActs["Test"][0] = list(chain.from_iterable(vec))
            #load training list
            imList1 = [os.path.basename(x) for x in glob.glob("Stim/Training/" + KN[kk][0] + "/*.jpg")] #pull image category list
            
            for jj in range(kk+1, len(KN)):
                #pull second image                       
                if KN[kk][2] == KN[jj][2]: #Check if they are matched for animacy              
                    
                    IM2 = image_loader("Stim/Test/" + cond[ii] + "/Obj (" + str(KN[jj][1]) + ")" + suf[ii] + ".png")    
                    
                    vec = model(IM2).detach().numpy() #Extract image vector
                    AllActs["Test"][1] = list(chain.from_iterable(vec))
                    
                    imList2 = [os.path.basename(x) for x in glob.glob("Stim/Training/" + KN[jj][0] + "/*.jpg")]
                    
                    tempScore = 0 
                    for fl in range(0,folK): #loop through folds
                        #randomize order of image lists every iteration
                        imList1 = random.sample(imList1, len(imList1))
                        imList2 = random.sample(imList2, len(imList2))
                        for tr in range(0,trK): #loop through training images
                            try:
                                trIM1= image_loader("Stim/Training/" + KN[kk][0] + "/" + imList1[tr])
                                trIM2= image_loader("Stim/Training/" + KN[jj][0] + "/" + imList2[tr])
                                
                                #Extract features for each training image
                                trVec1 = model(trIM1).detach().numpy() #Extract image vector
                                trVec2 = model(trIM2).detach().numpy() #Extract image vector
                                                                
                                #Add to dict
                                
                            except: #If there is some error with an image try next image outside the set
                                trIM1= image_loader("Stim/Training/" + KN[kk][0] + "/" + imList1[tr+trK]) #add number of training images to try out of set image
                                trIM2= image_loader("Stim/Training/" + KN[jj][0] + "/" + imList2[tr+trK])
                                
                                #Extract features for each training image
                                trVec1 = model(trIM1).detach().numpy() #Extract image vector
                                trVec2 = model(trIM2).detach().numpy() #Extract image vector
                            
                            AllActs["Train"][tr] = list(chain.from_iterable(trVec1))
                            AllActs["Train"][tr+trK] = list(chain.from_iterable(trVec2))
                        #Run SVM
                        clf = svm.SVC(kernel='linear', C=1).fit(AllActs["Train"], train_labels)
                        #Add current score to existing
                        tempScore = tempScore + clf.score(AllActs["Test"], test_labels)
                    
                    CNN_Acc[n,0] = KN[kk][0]
                    CNN_Acc[n,1] = KN[jj][0]
                    CNN_Acc[n,ii+2] = tempScore/folK
                    
                    print(np.round((n/210)*100,decimals = 2), models[mm], KN[kk][0], KN[jj][0], tempScore/folK)
                    n = n +1
                    
                else: #move to next iteration
                    continue
    
        np.savetxt('Results/' + models[mm] + '_SVM.csv', CNN_Acc, delimiter=',', fmt= '%s')
                
            
        

