curr_dir = '/user_data/vayzenbe/GitHub_Repos/kornet'
import pandas as pd
import numpy as np
import sys
vone_dir = '/user_data/vayzenbe/GitHub_Repos/vonenet'
cornet_dir = '/user_data/vayzenbe/GitHub_Repos/CORnet'
vit_dir = '/user_data/vayzenbe/GitHub_Repos/Cream/EfficientViT'
sys.path.insert(1, curr_dir)
sys.path.insert(1, vone_dir)
sys.path.insert(1, cornet_dir)
sys.path.insert(1, vit_dir)
import torch.nn as nn
import torch
from classification.model.build import EfficientViT_M0
import vonenet
import pdb




class TwoStream(nn.Module):
    def __init__(self):
        super(TwoStream, self).__init__()

        #define dorsal model
        dorsal = EfficientViT_M0()
        dorsal.head = nn.BatchNorm1d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        

        #define ventral model
        ventral = vonenet.get_model(model_arch='cornets_ff', pretrained=False).module
        ventral.model.decoder = nn.AdaptiveAvgPool2d(output_size=1)


        self.dorsal = dorsal
        self.ventral = ventral
        self.classifier = nn.Linear(in_features=704, out_features=565, bias=True)
        
    def forward(self, x1, x2):
        x1 = self.ventral(x1)
        x1 = torch.flatten(x1, 1)
        
        x2 = self.dorsal(x2)
        
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.classifier(x)
        return x