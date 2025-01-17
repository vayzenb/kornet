'''
Go through the acts and results folders and remove old files for models that were not used
'''

project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys

#add git_dir to path
sys.path.append(git_dir)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from glob import glob as glob
import pdb
import warnings
warnings.filterwarnings("ignore")

model_arch = ['vonenet_ff_ecoset','vonenet_ff_stylized-ecoset','vonenet_r_ecoset','vonenet_r_stylized-ecoset', 
              'SayCam', 'cvcl',
                'convnext', 'vit', 'vit_dinov2','clip_vit',
                'resnet50', 'resnet50_imagenet-sketch','resnet50_21k', 
                'resnet50_dino','clip_resnet_15m', 'clip_resnet']

results_dir = f'{git_dir}/results/models'
acts_dir = f'{git_dir}/modelling/acts'

#glob all the files in the results directory
results_files = glob(f'{results_dir}/*.csv')

#remove files that are not in the model arch
for file in results_files:
    file_name = file.split('/')[-1]
    #loop through model_arch and test if model is in the file name
    if not any([model in file_name for model in model_arch]):
        os.remove(file)
        print(f'{file_name} removed')


#do the same for the acts directory
acts_files = glob(f'{acts_dir}/*.npy')
for file in acts_files:
    file_name = file.split('/')[-1]
    #loop through model_arch and test if model is in the file name
    if not any([model in file_name for model in model_arch]):
        os.remove(file)
        print(f'{file_name} removed')
