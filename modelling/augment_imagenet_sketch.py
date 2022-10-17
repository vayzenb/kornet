import os
import shutil
from glob import glob as glob
import random as random
import pdb

image_dir = f'/lab_data/behrmannlab/image_sets'

target_set = f'{image_dir}/imagenet_sketch'
imagenet_dir = f'/lab_data/tarrlab/common/datasets/ILSVRC/Data/CLS-LOC'

n_ims = 10 #how many images to move to val

def create_vals(n_ims):
    '''
    create train and val
    '''
    print('creating sketch vals')
    folders = glob(f'{target_set}/train/*')
    os.makedirs(f'{target_set}/val', exist_ok=True)

    for folder in folders:
        images = glob(f'{folder}/*')
        curr_folder = folder.split('/')[-1]
        os.makedirs(f'{target_set}/val/{curr_folder}', exist_ok=True)
        
        random.shuffle(images)

        for image in images[:n_ims]:
            
            shutil.move(image, f'{target_set}/val/{curr_folder}')
            #print(image,f'{target_set}/val/{folder}')
        
def add_imagenet_vals(n_ims):
    print('adding imagenet vals')
    folders = glob(f'{imagenet_dir}/val/*/')
    
    for folder in folders:
        curr_folder = folder.split('/')[-2]
        os.makedirs(f'{target_set}/val/{curr_folder}', exist_ok=True)
        image_n = glob(f'{target_set}/val/{curr_folder}/*')
        
        if len(image_n) < n_ims*2:
            
            images = glob(f'{folder}/*')
            random.shuffle(images)
            for image in images[:n_ims]:
                
                shutil.copy(image, f'{target_set}/val/{curr_folder}')
        else:
            continue

def add_imagenet_trains(n_ims):
    print('adding imagenet trains')
    folders = glob(f'{imagenet_dir}/train/*/')

    for folder in folders:
        curr_folder = folder.split('/')[-2]
        os.makedirs(f'{target_set}/train/{curr_folder}', exist_ok=True)
        image_n = glob(f'{target_set}/train/{curr_folder}/*')
        
        if len(image_n) < n_ims*2:
            images = glob(f'{folder}/*')
            random.shuffle(images)
            for image in images[:n_ims]:
                
                shutil.copy(image, f'{target_set}/train/{curr_folder}')
        else:
            continue

#create_vals(10)
add_imagenet_vals(10)
add_imagenet_trains(40)


        



