'''
Extract acts for vision language models
'''
project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
sys.path.append(git_dir)


vone_dir = f'{cwd.split(project_name)[0]}vonenet'
#cornet_dir = '/user_data/vayzenbe/GitHub_Repos/CORnet'
vit_dir = f'{cwd.split(project_name)[0]}Cream/EfficientViT'
baby_dir = f'{cwd.split(project_name)[0]}multimodal-baby'
sys.path.insert(1, git_dir)
sys.path.insert(1, vone_dir)
#sys.path.insert(1, cornet_dir)
sys.path.insert(1, vit_dir)
sys.path.insert(1, baby_dir)
import pandas as pd
import numpy as np
import sys



import torch
import modelling.load_stim as load_stim
from glob import glob as glob
import pdb

#from model_loader import load_model as load_model

import clip
from PIL import Image
import numpy as np

import pandas as pd
from multimodal.multimodal_lit import MultiModalLitModel

print('libraries loaded...')

model_arch = sys.argv[1]
model_name = model_arch
stim_dir = sys.argv[2]

device = "cuda" if torch.cuda.is_available() else "cpu"

if model_arch == 'clip':
    model, transform = clip.load("ViT-B/32")

elif model_arch == 'cvcl':
    model, transform = MultiModalLitModel.load_model(model_name="cvcl")

model = model.to(device)
model.eval()




'''
#specify weights file
if len(sys.argv) == 2:
    weights = None
    model_name = model_arch
elif len(sys.argv) == 3:
    weights = sys.argv[2]
    model_name = model_arch + '_' + weights
'''

    

stim_folder = glob(f'{stim_dir}/*')
#only keep folder with bicycle
#stim_folder = [x for x in stim_folder if 'bicycle' in x]

suf = ''




def extract_acts(model, image_dir, transform):
    print('extracting features...')





    #Iterate through each image and extract activations

    imNum = 0
    n=0

    

    
    test_dataset = load_stim.load_stim(image_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers = 4, pin_memory=True)
    


    with torch.no_grad():
        
        for data, label in testloader:
            #print(label)
            # move tensors to GPU if CUDA is available
            
            data= data.cuda()
            
            _model_feats = model.encode_image(data)
            
            
            #output = model(data)
            
            out = np.vstack(_model_feats.cpu().numpy())
            

            if n == 0:
                acts = out
                #label_list = label
            else:
                acts= np.append(acts, out,axis = 0)
                #label_list = np.append(label_list, label)
                
            
            n = n + 1

    return acts






for cat_dir in stim_folder:
    print(cat_dir)
    #VIT runs out of memory quickly, so we delete and reload it after every iteration


    
    cat_name = cat_dir.split('/')[-1]
    print(model_arch, cat_name)
    acts = extract_acts(model, cat_dir, transform)

    

    
    
    np.save(f'{git_dir}/modelling/acts/{model_name}{suf}_{cat_name}.npy', acts)
    #clear memory
    del acts
    
    #clear cache
    torch.cuda.empty_cache()

    
