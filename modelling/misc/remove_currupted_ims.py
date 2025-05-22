import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import load_stim
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import tqdm
from glob import glob as glob
import natsort
import os

#measure time of script
import time

norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]

transform  = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.GaussianBlur(1, sigma=1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=norm_mean, std=norm_std)])


'''
Test how fast the dataloader is with different number of workers and pin_memory
'''
num_workers = np.arange(1,9).tolist()
pins = [False, True]

for pin in pins:
    for num_worker in num_workers:
                

        dataset = torchvision.datasets.ImageFolder('/lab_data/behrmannlab/image_sets/stylized-ecoset/val',transform=transform)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False,num_workers = num_worker, pin_memory=pin)
        #start time
        start = time.time()
        for data, label in trainloader:
            #print(len(data), len(label))
            continue

        #print end time in seconds
        end = time.time()
        print(f'num_workers: {num_worker}, pin_memory: {pin}, time: {end-start}')

'''
dataset = load_stim.load_stim('/lab_data/behrmannlab/image_sets/stylized-ecoset/train',transform=transform)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False,num_workers = 4, pin_memory=True)

folders = natsort.natsorted(glob('/lab_data/behrmannlab/image_sets/stylized-ecoset/train/*/'))
#start time
start = time.time()

n = 1
for folder in folders:
    dataset = load_stim.load_stim(folder,transform=transform)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False,num_workers = 4, pin_memory=True)

    print(n, len(folders))
    n = n + 1
    try:
        for data, label in trainloader:
            #print(len(data), len(label))
            continue
            
    except:
        print('Error in: ', folder)
        im_files = natsort.natsorted(glob(f'{folder}/*'))

        for im in im_files:
            try:
                image = Image.open(im).convert("RGB")
            except:
                print('error for', im)

                #delete im
                os.remove(im)

#end time
end = time.time()
print('time elapsed: ', end - start)
'''