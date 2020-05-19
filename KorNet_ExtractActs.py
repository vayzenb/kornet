# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:07:23 2020

Using the DiCarlo CORNet model (for now) extract acts for all training and test images 

@author: vayze
"""

#os.chdir('C:/Users/vayze/Desktop/GitHub_Repos/KorNet/')
import subprocess
import tqdm
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image
from itertools import chain
import pandas as pd

modelType = ['CorNet','FF_IN', 'R_IN', 'FF_SN', 'R_SN']
modelType = ['CorNet']
cond = ['Outline', 'Pert', 'IC']

suf = ['', '_ripple', '_IC']
maxTS = 5#Time steps

#Labels for training images
KN=pd.read_csv('KN_Classes.csv', sep=',',header=None).to_numpy() 

doTrain = True
doTest = True


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

for mm in range(0, len(modelType)):
    
    if modelType[mm] != 'CorNet':
        #select model to run
        if modelType[mm] == 'FF_IN':
            model = torchvision.models.alexnet(pretrained=True)
            new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
            model.classifier = new_classifier #replace model classifier with stripped version
            layer = "fc7"
            actNum = 4096
            
        elif modelType[mm] == 'R_IN':
            model = torchvision.models.resnet50(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            layer = "avgpool"
            actNum = 2048
                    
        elif modelType[mm] == 'FF_SN':
            model = torchvision.models.alexnet(pretrained=False)
            model.features = torch.nn.DataParallel(model.features)
            checkpoint = torch.load('ShapeNet_AlexNet_Weights.pth.tar')
            model.load_state_dict(checkpoint)
            new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
            model.classifier = new_classifier #replace model classifier with stripped version
            #model.to(device)
            layer = "fc7"
            actNum = 4096
            
        elif modelType[mm] == 'R_SN':
            model = torchvision.models.resnet50(pretrained=False)
            #model = torch.nn.DataParallel(model.features)
            checkpoint = torch.load('ShapeNet_ResNet50_Weights.pth.tar')
            model.load_state_dict(checkpoint)
            model = nn.Sequential(*list(model.children())[:-1])
            #model.to(device)
            layer = "avgpool"
            actNum = 2048
                       
        model.cuda()
        model.eval() #Set model into evaluation mode


    if doTest == True:
    ##########
    #Extract Acts for test images
    ##########
        for cc in range(0, len(cond)):
            print(modelType[mm], cond[cc])
            if modelType[mm] == 'CorNet': #Run if CorNet model
                for tS in range(1,maxTS+1):
                    #I'm not totally confident that the timesteps are different, but will evaluate later
                    CNet_cmd = "--data_path Stim/Test/" + cond[cc] + "/*.png --times " + str(tS) + " -o Activations/Test --ngpus 1 --outname " + cond[cc]
                    proc = subprocess.Popen("python run_cornet.py test --model RT " + CNet_cmd)
                    print(proc.wait())
                    #python run_cornet.py test --model S --data_path stim/test/IC --times 5 -o Activations --ngpus 1 --outname IC
                    #python run_cornet.py test --model S --data_path Stim/Test/IC --times 4 -o Activations --ngpus 1 --outname IC'
            else: #Non-CorNet models
                
                
                fnames = sorted(glob.glob(os.path.join("Stim/Test/" + cond[cc], '*.png')))
                allActs = np.zeros((len(fnames), actNum))
                n = 0
                for fname in tqdm.tqdm(fnames):
            
                    IM = image_loader(fname)
                    IM = IM.cuda()
            
            
                    vec = model(IM).cpu().detach().numpy()
                    vec =  np.reshape(vec, (len(vec), -1))
                    allActs[n,:] = vec
                    n = n+ 1
                    
                    fname = 'Activations/Test/' + modelType[mm] + '_' + cond[cc] + "_acts.npy"
                    np.save(fname, allActs)
                    
    if doTrain == True:           
    ####################
    #Extract Acts for train images
    ###################
        for kk in range(0, len(KN)):
            print(modelType[mm], KN[kk][0])
            
            if modelType[mm] == 'CorNet': #Run if CorNet model
                for tS in range(1,maxTS+1):
                    #I'm not totally confident that the timesteps are different, but will evaluate later
                    CNet_cmd = "--data_path Stim/Train/" + KN[kk][0] + "/*.jpg --times " + str(tS) + " -o Activations/Train --ngpus 1 --outname " + KN[kk][0]
                    
                    proc = subprocess.Popen("python run_cornet.py test --model RT " + CNet_cmd)
                    print(proc.wait())
                
            else: #Non-CorNet models
                
                fnames = sorted(glob.glob(os.path.join("Stim/Train/" + KN[kk][0], '*.jpg')))
                allActs = np.zeros((len(fnames), actNum))
                n = 0
                for fname in tqdm.tqdm(fnames):
            
                    IM = image_loader(fname)
                    IM = IM.cuda()
            
            
                    vec = model(IM).cpu().detach().numpy()
                    vec =  np.reshape(vec, (len(vec), -1))
                    allActs[n,:] = vec
                    n = n+ 1
                    
                    fname = 'Activations/Train/' + modelType[mm] + '_' + KN[kk][0] + "_acts.npy"
                    np.save(fname, allActs)
    
            
        
    
            
        
        