# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:10:34 2019

Iterates through each KorNet condition and provide top-5 labels

@author: VAYZENB
"""

import os
import sys
from collections import OrderedDict
import torch
import torchvision
import torchvision.models
from torch.autograd import Variable
from torch.utils import model_zoo
from PIL import Image
from skimage import io, transform
from skimage.viewer import ImageViewer
import numpy as np
import torchvision.transforms as T
import pandas as pd
import itertools

os.chdir('C:/Users/vayzenb/Desktop/GitHub Repos/KorNet/')

IMName = "Stim/Training/Dog/Dog_4 (2).jpg"
imsize = 224
loader = T.Compose([T.Resize(imsize), T.ToTensor()])



device = torch.device("cpu")

cond = ['Outline', 'Pert', 'IC',  'Outline_Black', 'Pert_Black', 'IC_Black', 'Outline_Black_Filled', 'Pert_Black_Filled']
#cond = ['Outline', 'Pert', 'IC']
suf = ['', '_ripple', '_IC', '', '_ripple', '_IC','', '_ripple']
#suf = ['', '_ripple', '_IC', ]
#Load ImageNet and KorNet classes
IN=pd.read_csv('IN_Classes.csv', sep=',',header=None).to_numpy()
KN=pd.read_csv('KN_Classes.csv', sep=',',header=None).to_numpy() 

#Load shape net
def load_model():

    model_weghts = "ShapeNet_Weights.pth.tar"

    model = torchvision.models.resnet50(pretrained=False)
    #model = torch.nn.DataParallel(model).cuda()
    #checkpoint = model_zoo.load_url(model_weghts)
    checkpoint = torch.load(model_weghts)
    model.load_state_dict(checkpoint)
    print("Using the ResNet50 architecture.")
    return model

scaler = T.Resize((224, 224))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = T.ToTensor()
#Set image loader for model
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0))
    return image  

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
for ii in range(0, len(KN)):
    CNN_Labels = np.empty((len(KN), 6), dtype = object)
    CNN_Labels[:,0] = KN[:,0]    
    
    #Loop through each image in a condition and predict class label
    for kk in range(0, len(KN)):
        imFile = 'Stim/Test/' + cond[ii] + '/OBJ (' + str(KN[kk,1]) + ')' + suf[ii] +'.png'
        IM = image_loader(imFile)

        out = predict(IM, model)
        out = out.numpy()
        out = out.tolist()
        out = list(itertools.chain(*out))
                
        CNN_Labels[kk,1:] = [IN[out[0]][0], IN[out[1]][0], IN[out[2]][0], IN[out[3]][0], IN[out[4]][0]]
        
    np.savetxt('Results/' + cond[ii] + "_PredClasses.csv", CNN_Labels, delimiter=',', fmt= '%s')

    

