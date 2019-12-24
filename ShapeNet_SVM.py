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


os.chdir('C:/Users/vayze/Desktop/GitHub Repos/KorNet/')


df=pd.read_csv('IN_Classes.csv', sep=',',header=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():

    model_weghts = "ShapeNet_Weights.pth.tar"

    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    #checkpoint = model_zoo.load_url(model_weghts)
    checkpoint = torch.load(model_weghts)
    model.load_state_dict(checkpoint["state_dict"])
    print("Using the ResNet50 architecture.")
    return model

imsize = 224
IMName = "C:/Users/vayze/Desktop/GitHub Repos/KorNet/Stim/Test/Outline/OBJ (1).png"
IMName = "Stim/Training/Dog/Dog_4 (2).jpg"
loader = T.Compose([T.Resize(imsize), T.ToTensor()])
model = load_model(model_C) 

model.eval()
    
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU    


def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return classes.item()

label = predict(IM, model)

df.iat[label.item()-1,0]
