import sys
curr_dir = '/user_data/vayzenbe/GitHub_Repos/kornet'

#add curr_dir to working directory
sys.path.append(curr_dir)
import pandas as pd
import numpy as np
import random
import os
import shutil
from glob import glob as glob
import pdb


source_dir = f'/user_data/vayzenbe/image_sets/ecoset'
target_dir = f'/user_data/vayzenbe/image_sets/kornet_images'
classlist = pd.read_csv(f'{curr_dir}/stim/kornet_train_sources.csv')


im_num = 250 #number of images to copy

for obj_class in classlist['object']:
    #glob image folder
    im_list = glob(f'{source_dir}/train/*_{obj_class}/*.JPEG')
    #check if folder exists


    #check if im_list is empty or folder exists
    if len(im_list) == 0 or os.path.exists(f'{target_dir}/{obj_class}'):
        print(f'No images found for {obj_class} or they are already copied')

    else:
        
        print('Copying images for: ' + obj_class)

        #make target folder
        os.makedirs(f'{target_dir}/{obj_class}', exist_ok=True)        

        #shuffle image list
        random.shuffle(im_list)
        #pdb.set_trace()
        #copy first N objects
        for im in im_list[:im_num]:
            #copy image to target
            shutil.copy(im, f'{target_dir}/{obj_class}')
            


        
     

        

