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



target_dir = f'/user_data/vayzenbe/image_sets/kornet_images'
classlist = pd.read_csv(f'{curr_dir}/stim/kornet_train_sources.csv')


im_num = 500 #number of images to copy

for obj_class, source in zip(classlist['object'], classlist['source1']):
    im_num = 500
    if source == 'ecoset':
        source_dir = f'/user_data/vayzenbe/image_sets/ecoset'
        #glob image folder
        im_list = glob(f'{source_dir}/train/*{obj_class}/*')
        
    elif source == 'imagenet21k':
        source_dir = f'/user_data/vayzenbe/image_sets/imagenet21k'
        im_list = glob(f'{source_dir}_subset/{obj_class}/*')
        #pdb.set_trace()

    




    
    
    #check number of images in target folder
    if os.path.exists(f'{target_dir}/{obj_class}'):
        target_num = len(glob(f'{target_dir}/{obj_class}/*'))
        if target_num >= im_num:
            print(f'{obj_class} has {target_num} images, skipping')
            continue
        else:
            im_num = im_num - target_num


    #check if im_list is empty or folder exists
    if len(im_list) == 0:
        print(f'No images found for {obj_class}')

    
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
            


        
     

        

