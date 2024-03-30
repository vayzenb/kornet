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




class MyEnsemble(nn.Module):
    def __init__(self):
        super(MyEnsemble, self).__init__()

        #define dorsal model
        dorsal = EfficientViT_M0()
        classifier = nn.Sequential(*list(dorsal.children())[-1])[:-1]
        dorsal = nn.Sequential(*list(dorsal.children())[:-1])
        #recomvbine model and classifier
        dorsal = nn.Sequential(dorsal, classifier)
        print(dorsal)

        #define ventral model
        ventral = vonenet.get_model(model_arch='cornets_ff', pretrained=False).module
        ventral = nn.Sequential(*list(ventral.children())[:-1], nn.Sequential(*list(ventral.model.children())[:-1]),nn.AdaptiveAvgPool2d(output_size=1))


        self.dorsal = dorsal
        self.ventral = ventral
        self.classifier = nn.Linear(in_features=704, out_features=565, bias=True)
        
    def forward(self, x1, x2):
        x1 = self.ventral(x1)
        print('ventral out', x1.shape)
        x2 = self.dorsal(x2)
        
        print('dorsal out', x2.shape)
        x = torch.cat((x1, x2), dim=1)
        print('concat out', x.shape)
        x = self.classifier(x)
        return x