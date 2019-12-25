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


#Load ImageNet and KorNet classes
IN=pd.read_csv('IN_Classes.csv', sep=',',header=None).to_numpy()
KN=pd.read_csv('KN_Classes.csv', sep=',',header=None).to_numpy() 

#Load shape net
def load_model():

    model_weights = "ShapeNet_Weights.pth.tar"

    model = torchvision.models.resnet50(pretrained=False)
    #model = torch.nn.DataParallel(model).cuda()
    #checkpoint = model_zoo.load_url(model_weghts)
    checkpoint = torch.load(model_weights)
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

def get_vector(image_name):
    
    my_embedding = torch.zeros([1,2048])
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(-1))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    t_img = image_loader(image_name)
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


model = load_model() 
model.eval() #Set model into evaluation mode
layer = model._modules.get('avgpool')

