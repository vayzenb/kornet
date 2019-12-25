# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:59:59 2019

@author: VAYZENB
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torchsummary import summary
import os

os.chdir('C:/Users/vayze/Desktop/GitHub Repos/KorNet/')

pic_one = str("Stim/Training/Dog/Dog_4 (2).jpg")
pic_two = str("Stim/Training/All/Dog/Dog_4 (2).jpg")

# Load the pretrained model
model = models.resnet50(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
#model = model.to(device)
#model = model.to("cpu")
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

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



# Set model to evaluation mode
model.eval()

#Image transforms
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros([1,2048])
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(-1))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

pic_one_vector = get_vector(pic_one)
vec = pic_one_vector.numpy()
#pic_two_vector = get_vector(pic_two)