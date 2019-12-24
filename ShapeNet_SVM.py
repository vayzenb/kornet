# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:10:34 2019

@author: VAYZENB
"""

import os
import sys
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as T
from torch.utils import model_zoo
from PIL import Image
from skimage import io, transform
from skimage.viewer import ImageViewer
import numpy as np

import pandas as pd



os.chdir('C:/Users/vayze/Desktop/GitHub Repos/KorNet/')

IMName = "Stim/Training/Dog/Dog_4 (2).jpg"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cond = ['Outline', 'Pert', 'IC',  'Silh_Black', 'Pert_White', 'Pert_Black']
suf = ['', '_ripple', '_IC', '', '_ripple']
#Load ImageNet and KorNet classes
IN=pd.read_csv('IN_Classes.csv', sep=',',header=None).to_numpy()
KN=pd.read_csv('KN_Classes.csv', sep=',',header=None).to_numpy() 

#Load shape net
def load_model():

    model_weghts = "ShapeNet_Weights.pth.tar"

    model = models.resnet50(pretrained=False)
    model = nn.DataParallel(model)
    #checkpoint = model_zoo.load_url(moel_weghts)
    checkpoint = torch.load(model_weghts)
    model.load_state_dict(checkpoint["state_dict"])
    #model = nn.Sequential(*list(model.children())[:-2])
    
    print("Using the ResNet50 architecture.")
    return model

scaler = T.Resize((224, 224))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = T.ToTensor()
def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

imsize = 224
loader = T.Compose([T.Resize(imsize), T.ToTensor()])
#Might have to integrate this normalize function at some point

##Set image loader for model
#def image_loader(image_name):
#    """load image, returns cuda tensor"""
#    image = Image.open(image_name)
#    image = loader(image).float()
#    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#    return image.cuda()  #assumes that you're using GPU    


#res50_model = models.resnet50(pretrained=True)
#res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
model = load_model() 
#layer = model._modules.get('avgpool')
#model.eval() #Set model into evaluation mode





