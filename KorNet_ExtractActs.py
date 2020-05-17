# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:07:23 2020

Using the DiCarlo CORNet model (for now) extract acts for all training and test images 

@author: vayze
"""


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

modelType = ['CorNet','FF_IN', 'R_IN', 'FF_SN', 'R_SN']
modelType = ['CorNet']
cond = ['Outline', 'Pert', 'IC']

suf = ['', '_ripple', '_IC']
maxTS = 5#Time steps

#Load acts for test images


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

    for cc in range(0, len(cond)):
        print(modelType[mm], cond[cc])
        if modelType[mm] == 'CorNet':
            for tS in range(1,maxTS):
                #I'm not totally confident that the timesteps are different, but will evaluate later
                CNet_cmd = "--data_path Stim/Test/" + cond[cc] + " --times " + str(tS) + " -o Activations --ngpus 1 --outname " + cond[cc]
                print(CNet_cmd)
                subprocess.Popen("python run_cornet.py test --model S " + CNet_cmd)
        else:
            
            
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
                
                fname = model[mm] + '_' + cond[cc] + "_acts.npy"
                np.save(fname, allActs)
            

            
    
            
        
        