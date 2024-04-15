"""
Dataloader which loads images with the folder as the label
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import natsort
from PIL import Image


class load_stim(Dataset):

    def __init__(self, main_dir, transform_ventral=None, transform_dorsal=None):
        self.main_dir = main_dir
        self.transform_ventral = transform_ventral
        self.transform_dorsal = transform_dorsal
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        


    def __len__(self):
        return len(self.total_imgs)
    
    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        
        image = Image.open(img_loc).convert("RGB")
        #image = cv2.imread(img_loc)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_ventral = self.transform_ventral(image)
        image_dorsal = self.transform_dorsal(image)

        return image_ventral,image_dorsal, self.total_imgs[idx]
