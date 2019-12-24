# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:10:34 2019

@author: VAYZENB
"""

import os
import sys
from collections import OrderedDict
import torch
import torchvision
import torchvision.models
from torch.utils import model_zoo
from PIL import Image
from skimage import io, transform
from skimage.viewer import ImageViewer
import numpy as np
import torchvision.transforms as T
import pandas as pd
import itertools

os.chdir('C:/Users/vayze/Desktop/GitHub Repos/KorNet/')

IMName = "Stim/Training/Dog/Dog_4 (2).jpg"
imsize = 224
loader = T.Compose([T.Resize(imsize), T.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cond = ['Outline', 'Pert', 'IC',  'Silh_Black', 'Pert_White', 'Pert_Black']
suf = ['', '_ripple', '_IC', '', '_ripple']
#Load ImageNet and KorNet classes
IN=pd.read_csv('IN_Classes.csv', sep=',',header=None).to_numpy()
KN=pd.read_csv('KN_Classes.csv', sep=',',header=None).to_numpy() 

#Load shape net
def load_model():

    model_weghts = "ShapeNet_Weights.pth.tar"

    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    #checkpoint = model_zoo.load_url(model_weghts)
    checkpoint = torch.load(model_weghts)
    model.load_state_dict(checkpoint["state_dict"])
    print("Using the ResNet50 architecture.")
    return model

#Set image loader for model
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU    

#Predict image class
def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(5, dim=1)
    return classes


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
        out = out.cpu().detach().numpy()
        out = out.tolist()
        out = list(itertools.chain(*out))
                
        CNN_Labels[kk,1:] = [IN[out[0]][0], IN[out[1]][0], IN[out[2]][0], IN[out[3]][0], IN[out[4]][0]]
        
    np.savetxt('Results/' + cond[ii] + "_PredClasses.csv", CNN_Labels, delimiter=',', fmt= '%s')

    

